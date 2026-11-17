"""
KidneyScan AI — Training (KiTS23)
==================================
Training loop for the light 3D U-Net.

Data layout (KiTS23, after download — see the plan):
    Official layout:
      data/
        case_00000/imaging.nii.gz
        case_00000/segmentation.nii.gz
        ... (489 training cases, case_00000 .. case_00488)
    Lite layout (YongchengYAO/KiTS23-Lite, 40.9 GB — the Kaggle default):
      data/
        Images/<case>.nii.gz
        Masks/<case>.nii.gz
    Both layouts are accepted: list_cases() finds pairs in either one
    (same pairing logic as data-prep/check_data.py). nnU-Net layout
    (imagesTr/<case>_0000.nii.gz + labelsTr/<case>.nii.gz) is accepted too.
    Labels: 0 background, 1 kidney, 2 tumor, 3 cyst.

Split (fixed, so every ablation run is comparable — see ablation_plan.md):
    400 train / 89 validation, from the 489 public training cases.
    The 110 hidden KiTS23 test cases are NEVER used for training.
    Split seed is fixed (42) and the split file is saved, so a rerun
    uses exactly the same cases.

Phases (flags, not separate scripts):
    --phase phase1 : 20 cases (16 train / 4 val), 30 epochs, ~2-3 GPU hours.
                     Stop here if Dice is near zero — fix data first.
    --phase phase2 : full 400/89 split, 50-100 epochs, ~12-20 GPU hours,
                     split across Kaggle sessions with --resume.

Preprocessing (nnU-Net-style, simplified — see REUSE.md):
    - Clip CT Hounsfield units to [-200, 300] (kidney window; soft tissue
      and contrast-enhanced kidney live in this range).
    - Normalise to ~[0, 1] from that window.
    - Crop to the kidney region (bounding box of the mask + margin) when
      a mask exists; at inference time crop to the body region instead.
    - Resize/pad patches to --patch-size (default 128x128x64).

Augmentation (on the fly, per batch):
    - Random flip on each axis, random 90-degree rotation in-plane,
      random zoom (0.9-1.1), small Gaussian noise. Applied to image AND
      mask together so they stay aligned.

Checkpoints: saved every --save-every epochs and at the end, plus
``best.pt`` for the best validation mean Dice (kidney+tumor+cyst).
Resume with --resume <checkpoint.pt> — optimizer + epoch are restored,
so a Kaggle session ending mid-run loses at most a few epochs.

Mixed precision (AMP): on by default on GPU, off on CPU.
GPU memory: light model + patch 128x128x64 + batch 2 fits a Kaggle
free GPU (T4/P100, 16GB). If it OOMs, drop to --patch-size 96 96 64
or --batch-size 1.

Run on Kaggle (PyTorch preinstalled there). On this VM PyTorch may not
be installed — see the note at the bottom of this file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:  # PyTorch not installed on this VM — code still imports.
    torch = None
    Dataset = object  # placeholder so the class definition below still parses
    TORCH_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nib = None
    NIBABEL_AVAILABLE = False

try:
    from light_unet3d import LightUNet3D, NUM_CLASSES, count_parameters
    from losses import KidneyLoss, DEFAULT_CLASS_WEIGHTS, compute_class_weights
except ImportError:  # torch missing — main() reports this cleanly
    LightUNet3D = None
    NUM_CLASSES = 4
    count_parameters = None
    KidneyLoss = None
    DEFAULT_CLASS_WEIGHTS = [0.1, 1.0, 3.0, 5.0]
    compute_class_weights = None

# Fixed split — every experiment uses the same cases.
SPLIT_SEED = 42
PHASE1_CASES = 20
PHASE1_EPOCHS = 30
PHASE2_TRAIN = 400
PHASE2_VAL = 89

CT_MIN, CT_MAX = -200.0, 300.0  # HU window for kidney CT


# ---------------------------------------------------------------- data ---

def load_nifti(path: Path):
    """Load a .nii.gz file. Returns (array float32, voxel spacing in mm)."""
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required to load .nii.gz files. "
                          "Install it on Kaggle with: pip install nibabel")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def preprocess_ct(ct: np.ndarray) -> np.ndarray:
    """Clip HU window and normalise to [0, 1]."""
    ct = np.clip(ct, CT_MIN, CT_MAX)
    return (ct - CT_MIN) / (CT_MAX - CT_MIN)


def crop_to_mask(ct: np.ndarray, mask: np.ndarray, margin: int = 10):
    """Crop image+mask to the bounding box of kidney/tumor/cyst + margin."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return ct, mask
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, np.array(ct.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    return ct[slices], mask[slices]


def crop_to_body_center(ct: np.ndarray, frac: float = 0.6,
                        hu_threshold: float = -500.0):
    """Crop to the central ``frac`` of the body bounding box (HU > threshold).

    Inference framing fix (2026-11-17): training crops every case to the
    kidney bbox first (``crop_to_mask``), so the model only ever saw tight
    kidney close-ups. Feeding it the whole abdomen at inference gave
    Dice 0.0 on a real case — the kidney was smeared and misplaced.
    The body bbox (air vs tissue) is computable without any ground
    truth, and its central 60% contains the kidneys in KiTS23 scans
    (they are kidney-centred volumes). Validated on 3 real cases with
    the live checkpoint (fg Dice): case_00009 0.67, case_00030 0.62,
    case_00038 0.70 — vs 0.0 for the whole-volume fast path and 0.01
    for the full body bbox. Returns (cropped_volume, slices) so the
    predicted mask can be placed back into the full volume.
    """
    body = ct > hu_threshold
    if not body.any():
        slices = tuple(slice(None) for _ in range(ct.ndim))
        return ct, slices
    coords = np.argwhere(body)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    cen = (lo + hi) / 2.0
    half = (hi - lo) * float(frac) / 2.0
    mins = np.maximum(np.round(cen - half).astype(int), 0)
    maxs = np.minimum(np.round(cen + half).astype(int),
                      np.array(ct.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    return ct[slices], slices


def resize_to_patch(vol: np.ndarray, patch, order: int = 1) -> np.ndarray:
    """Resize a 3D volume to the patch size (simple trilinear/nearest via torch)."""
    vol = np.ascontiguousarray(vol)  # flips/rot90 give negative strides
    t = torch.from_numpy(vol[None, None]).float()
    out = torch.nn.functional.interpolate(
        t, size=tuple(patch), mode="trilinear" if order == 1 else "nearest",
        align_corners=False if order == 1 else None)
    return out[0, 0].numpy()


class KiTS23Dataset(Dataset):
    """One case = one CT + mask, cropped and resized to a patch.

    Args:
        cases: list of (case_id, image_path, mask_path) tuples, as
            returned by list_cases(). Works for the official case-folder
            layout and the Lite Images/Masks layout.
        patch_size: (D, H, W) patch, e.g. (128, 128, 64).
        augment: apply random flip/rotate/zoom/noise.
        centred_sampling: with probability ``centred_prob`` take a
            patch at NATIVE resolution centred on a cyst voxel
            (cyst first, then tumor, then kidney) instead of
            resizing the whole volume — see sample_centred_patch().
            Default False = old whole-volume resize behaviour.
        centred_prob: probability of a centred patch per case
            when centred_sampling is on. Default 0.5.
    """

    def __init__(self, cases, patch_size=(128, 128, 64), augment=False,
                 centred_sampling=False, centred_prob=0.5):
        self.cases = [(str(cid), Path(img), Path(msk)) for cid, img, msk in cases]
        self.patch_size = tuple(patch_size)
        self.augment = augment
        self.centred_sampling = centred_sampling
        self.centred_prob = float(centred_prob)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        _case_id, img_path, mask_path = self.cases[idx]
        ct, _spacing = load_nifti(img_path)
        mask, _ = load_nifti(mask_path)
        ct = preprocess_ct(ct)
        ct, mask = crop_to_mask(ct, mask)
        if self.centred_sampling and np.random.rand() < self.centred_prob:
            slices = sample_centred_patch(mask, self.patch_size,
                                          rng=np.random)
            if slices is not None:
                # Native-resolution crop: the cyst keeps its real size
                # (an 8x8x8 cyst stays 512 voxels) instead of shrinking
                # to ~10 voxels under the whole-volume resize.
                ct = ct[slices]
                mask = mask[slices]
            else:
                ct = resize_to_patch(ct, self.patch_size, order=1)
                mask = resize_to_patch(mask.astype(np.float32),
                                       self.patch_size,
                                       order=0).round().astype(np.int64)
        else:
            ct = resize_to_patch(ct, self.patch_size, order=1)
            mask = resize_to_patch(mask.astype(np.float32), self.patch_size,
                                   order=0).round().astype(np.int64)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        if self.augment:
            ct, mask = augment_pair(ct, mask)
        x = torch.from_numpy(ct[None].astype(np.float32))
        y = torch.from_numpy(mask.astype(np.int64))
        return x, y


def sample_centred_patch(mask: np.ndarray, patch_size,
                         rng=None, priority_labels=(3, 2, 1)):
    """Crop window of ``patch_size`` centred on a foreground voxel.

    Cyst-centred sampling (2026-11-16): the old path resizes the
    WHOLE case to 128x128x64, so an 8x8x8 cyst survives as ~10
    voxels (2 voxels wide) — too small to learn from. This helper
    instead returns a window at NATIVE resolution centred on a
    cyst voxel (label 3 first, then tumor 2, then kidney 1), so
    the cyst keeps its full size in the patch.

    Args:
        mask: integer label volume (after crop_to_mask).
        patch_size: (D, H, W) window, e.g. (128, 128, 64).
        rng: random source with ``randint(low, high)`` (np.random
            module or np.random.RandomState). Default: np.random.
        priority_labels: labels tried in order; the first label
            present in the mask wins.
    Returns a tuple of 3 slices, or None when no centred window
    is possible (volume smaller than the patch in some dim, or
    no foreground voxel at all) — the caller then falls back to
    the whole-volume resize path.
    """
    if rng is None:
        rng = np.random
    patch = tuple(int(p) for p in patch_size)
    if any(s < p for s, p in zip(mask.shape, patch)):
        return None
    for label in priority_labels:
        coords = np.argwhere(mask == label)
        if len(coords) == 0:
            continue
        centre = coords[rng.randint(0, len(coords))]
        slices = []
        for dim in range(3):
            start = int(centre[dim]) - patch[dim] // 2
            start = max(0, min(start, mask.shape[dim] - patch[dim]))
            slices.append(slice(start, start + patch[dim]))
        return tuple(slices)
    return None


def augment_pair(ct: np.ndarray, mask: np.ndarray):
    """Flip / rotate / zoom / noise, applied to image and mask together."""
    rng = np.random
    for axis in range(3):
        if rng.rand() < 0.5:
            ct = np.flip(ct, axis=axis)
            mask = np.flip(mask, axis=axis)
    k = rng.randint(0, 4)  # 0, 90, 180, 270 degrees in the H-W plane
    if k:
        # FIX 2026-11-16 (found in Kaggle Phase 1): a 90/270-degree rot90
        # swaps H and W, so a 128x128x64 patch becomes 128x64x128 and the
        # DataLoader collate fails. Resize back to the original patch shape,
        # same as the zoom step below does.
        orig_shape = ct.shape
        ct = np.rot90(ct, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(1, 2))
        if ct.shape != orig_shape:
            ct = resize_to_patch(ct, orig_shape, order=1)
            mask = resize_to_patch(mask.astype(np.float32), orig_shape, order=0).round()
    zoom = rng.uniform(0.9, 1.1)
    if abs(zoom - 1.0) > 0.01:
        orig_shape = ct.shape
        ct = resize_to_patch(ct, [int(s * zoom) for s in orig_shape], order=1)
        ct = resize_to_patch(ct, orig_shape, order=1)  # back to patch size
        mask = resize_to_patch(mask.astype(np.float32),
                               [int(s * zoom) for s in orig_shape], order=0)
        mask = resize_to_patch(mask, orig_shape, order=0).round()
    if rng.rand() < 0.5:
        ct = ct + rng.normal(0, 0.02, size=ct.shape).astype(np.float32)
        ct = np.clip(ct, 0, 1)
    return np.ascontiguousarray(ct), np.ascontiguousarray(mask)


def _case_stem(filename: str) -> str:
    """Case id from a .nii.gz file name (strips .nii.gz and _0000)."""
    name = filename[:-len(".nii.gz")] if filename.endswith(".nii.gz") else filename
    if name.endswith("_0000"):
        name = name[:-len("_0000")]
    return name


def list_cases(data_dir: Path):
    """All (case_id, image_path, mask_path) pairs under data_dir.

    Accepts the official layout (case_XXXXX/imaging.nii.gz +
    segmentation.nii.gz), the Lite layout (Images/<case>.nii.gz +
    Masks/<case>.nii.gz) and the nnU-Net layout (imagesTr/<case>_0000
    .nii.gz + labelsTr/<case>.nii.gz). Sorted by case id, no duplicates.
    Cases with no mask file are skipped (training needs the mask).
    """
    data_dir = Path(data_dir)
    pairs = {}  # case_id -> (image_path, mask_path)

    # Official layout: case folders.
    if data_dir.is_dir():
        for entry in sorted(data_dir.iterdir()):
            if not entry.is_dir():
                continue
            img = entry / "imaging.nii.gz"
            msk = entry / "segmentation.nii.gz"
            if img.is_file() and msk.is_file():
                pairs[entry.name] = (img, msk)

    # Flat layouts: Images/Masks (Lite), imagesTr/labelsTr (nnU-Net),
    # images/masks (lowercase variant).
    for img_dir_name, msk_dir_name in (("Images", "Masks"),
                                       ("imagesTr", "labelsTr"),
                                       ("images", "masks")):
        img_dir = data_dir / img_dir_name
        msk_dir = data_dir / msk_dir_name
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.nii.gz")):
            stem = _case_stem(img.name)
            if stem in pairs:
                continue
            msk = None
            if msk_dir.is_dir():
                for cand in (stem + ".nii.gz", stem + "_0000.nii.gz"):
                    if (msk_dir / cand).is_file():
                        msk = msk_dir / cand
                        break
            if msk is not None:
                pairs[stem] = (img, msk)

    return [(cid, img, msk) for cid, (img, msk) in sorted(pairs.items())]


# ------------------------------------------------------- cyst oversample ---

def find_cyst_cases(cases, cache_file: Path | None = None):
    """Case ids whose mask contains at least one cyst voxel (label 3).

    Loads each mask once (masks only, not images) and, when a cache
    file is given, saves {case_id: true/false} as JSON so later runs
    skip re-reading the masks. A cached entry is trusted as-is; delete
    the cache file if the masks change. Deterministic: cases are
    checked in sorted case-id order, cache keys are sorted on write.
    Returns a set of case ids.
    """
    cases = sorted(cases, key=lambda c: c[0])
    cache = {}
    if cache_file is not None:
        cache_file = Path(cache_file)
        if cache_file.is_file():
            try:
                cache = json.loads(cache_file.read_text())
            except (json.JSONDecodeError, OSError):
                cache = {}
    cyst_ids = set()
    changed = False
    for cid, _img, msk in cases:
        if cid in cache:
            if cache[cid]:
                cyst_ids.add(cid)
            continue
        has_cyst = False
        try:
            mask, _ = load_nifti(Path(msk))
            has_cyst = bool((mask == 3).any())
        except Exception:  # noqa: BLE001 - unreadable mask: treat as no cyst
            has_cyst = False
        cache[cid] = has_cyst
        changed = True
        if has_cyst:
            cyst_ids.add(cid)
    if cache_file is not None and changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(
            {k: cache[k] for k in sorted(cache)}, indent=2))
    return cyst_ids


def apply_cyst_oversample(train_cases, repeat: int = 2,
                          cache_file: Path | None = None, seed: int = 42):
    """Duplicate cyst-containing cases in the training list.

    Cyst fix (2026-11-16): the dataset resizes the WHOLE case to
    128x128x64 with no patch sampling, so tiny cysts shrink to a few
    voxels and most batches contain almost no cyst signal. Duplicating
    cyst cases (repeat=2 means each cyst case appears twice per epoch)
    is the simplest fix that fits this whole-volume dataset — no
    patch sampler to rewrite. Deterministic: cyst ids are found in
    sorted order, extras are appended in sorted case-id order, and
    the DataLoader's own shuffle (seeded by the training loop) mixes
    them. ``seed`` is kept for future shuffling use and recorded in
    the log line. Returns (new_train_cases, n_cyst_cases).
    """
    if repeat < 2:
        return list(train_cases), 0
    _ = seed  # ordering is sorted, not random; seed kept for reproducibility
    cyst_ids = find_cyst_cases(train_cases, cache_file=cache_file)
    by_id = {c[0]: c for c in train_cases}
    extras = []
    for _ in range(repeat - 1):
        extras.extend(by_id[cid] for cid in sorted(cyst_ids) if cid in by_id)
    return list(train_cases) + extras, len(cyst_ids)


def make_split(cases, phase: str, split_file: Path | None = None):
    """Fixed 400/89 split (phase2) or 16/4 from 20 cases (phase1).

    The split file is written for BOTH phases, so a rerun uses exactly
    the same cases (previously phase1 returned early and never wrote it).
    """
    rng = random.Random(SPLIT_SEED)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    if phase == "phase1":
        subset = shuffled[:PHASE1_CASES]
        train, val = subset[:16], subset[16:20]
    else:
        train = shuffled[:PHASE2_TRAIN]
        val = shuffled[PHASE2_TRAIN:PHASE2_TRAIN + PHASE2_VAL]
    if split_file is not None:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps({
            "seed": SPLIT_SEED,
            "train": [c[0] for c in train],
            "val": [c[0] for c in val],
        }, indent=2))
    return train, val


# ------------------------------------------------------------- training ---

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler,
                    use_amp: bool):
    model.train()
    totals = {"total": 0.0, "dice": 0.0, "ce": 0.0, "edge": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss, parts = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss, parts = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
        for k in totals:
            totals[k] += float(parts[k])
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


def _no_grad_if_torch(fn):
    """torch.no_grad when torch exists, plain function otherwise."""
    if TORCH_AVAILABLE:
        return torch.no_grad()(fn)
    return fn


@_no_grad_if_torch
def validate(model, loader, loss_fn, device):
    model.eval()
    totals = {"total": 0.0, "dice": 0.0}
    per_class = np.zeros(NUM_CLASSES)
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        _loss, parts = loss_fn(logits, y)
        totals["total"] += float(parts["total"])
        totals["dice"] += float(parts["dice"])
        per_class += parts["per_class_dice"].cpu().numpy()
        n += 1
    per_class /= max(n, 1)
    # Mean Dice over foreground classes only (kidney, tumor, cyst).
    fg_dice = float(per_class[1:].mean())
    return {k: v / max(n, 1) for k, v in totals.items()}, per_class, fg_dice


def main():
    ap = argparse.ArgumentParser(description="Train KidneyScan light 3D U-Net")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Folder with case_XXXXX/imaging.nii.gz + segmentation.nii.gz")
    ap.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs (default: 30 phase1, 80 phase2)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 64])
    ap.add_argument("--base-channels", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    ap.add_argument("--num-workers", type=int, default=2)
    # Cyst fix flags (2026-11-16; all default OFF so old runs reproduce).
    ap.add_argument("--cyst-oversample", action="store_true",
                    help="Duplicate cyst-containing train cases (see "
                    "--cyst-oversample-repeat). Off by default.")
    ap.add_argument("--cyst-oversample-repeat", type=int, default=2,
                    help="Copies of each cyst case per epoch when "
                    "--cyst-oversample is on. Default: 2.")
    ap.add_argument("--cyst-cache", type=Path, default=None,
                    help="JSON cache for cyst presence per case (avoids "
                    "re-reading masks every run).")
    ap.add_argument("--cyst-tversky-weight", type=float, default=0.0,
                    help="Weight of the cyst Tversky loss (FN-heavy, "
                    "alpha 0.7 / beta 0.3). 0.0 = off (default). "
                    "Recommended ON value to try first: 1.0.")
    ap.add_argument("--extra-cyst-dice-weight", type=float, default=0.0,
                    help="Weight of the extra cyst-only Dice term. "
                    "0.0 = off (default). Recommended ON value: 1.0. "
                    "Try one cyst term at a time for a clean ablation.")
    ap.add_argument("--cap-cyst", type=float, default=10.0,
                    help="Max class weight for cyst only (tumor stays "
                    "capped at 5.0). Default: 10.0. Pass 5.0 to restore "
                    "the old shared cap.")
    # Cyst-centred patch sampling (2026-11-16; default OFF).
    ap.add_argument("--cyst-centred-sampling", action="store_true",
                    help="With probability --cyst-centred-prob, take a "
                    "native-resolution patch centred on a cyst voxel "
                    "(cyst first, then tumor, then kidney) instead of "
                    "resizing the whole volume. Off by default — the "
                    "old whole-volume resize path is unchanged. "
                    "Validation always uses the whole-volume path.")
    ap.add_argument("--cyst-centred-prob", type=float, default=0.5,
                    help="Probability of a centred patch per training "
                    "case when --cyst-centred-sampling is on. "
                    "Default: 0.5. Recommended first try: 0.5.")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "PyTorch is NOT installed in this environment. "
            "This code runs on Kaggle (PyTorch preinstalled) or after:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install nibabel\n"
            "See train_kaggle.py for the Kaggle-ready version.")
    if not NIBABEL_AVAILABLE:
        raise SystemExit("nibabel is required: pip install nibabel")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and device == "cuda"
    print(f"device={device} amp={use_amp} phase={args.phase}")

    cases = list_cases(args.data_dir)
    print(f"found {len(cases)} cases in {args.data_dir}")
    if args.phase == "phase1" and len(cases) < PHASE1_CASES:
        print(f"WARNING: phase1 expects {PHASE1_CASES} cases, only "
              f"{len(cases)} found — val split will be smaller than 4. "
              f"Download more cases before judging Dice.")
    train_cases, val_cases = make_split(
        cases, args.phase, split_file=args.out_dir / "split.json")
    print(f"train={len(train_cases)} val={len(val_cases)}")

    # Class weights from the real training data (not the fallback).
    # Counts voxels over ALL train cases (not just the first 20):
    # a 20-case sample can miss cysts almost entirely and give a
    # wrong cyst weight. Counting is one mask-only pass, still fast.
    # Counted BEFORE cyst oversampling, so duplicated cyst cases do
    # not skew the weights (oversampling changes sampling frequency,
    # not the data distribution the weights describe).
    try:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _case_id, _img_path, mask_path in train_cases:
            mask, _ = load_nifti(mask_path)
            for c in range(NUM_CLASSES):
                counts[c] += int((mask == c).sum())
        weights = compute_class_weights(
            counts, cap_cyst=args.cap_cyst).tolist()
        print(f"class weights from real data (bg, kidney, tumor, cyst): {weights}")
    except Exception as e:  # noqa: BLE001 - fall back, do not crash training
        print(f"could not count voxels ({e}); using fallback weights")
        weights = DEFAULT_CLASS_WEIGHTS

    # Cyst oversampling (opt-in, default off). Duplicates cyst cases in
    # the training list only — the val split is never touched. Done
    # AFTER the weight count above, so duplicates do not skew weights.
    if args.cyst_oversample:
        n_before = len(train_cases)
        train_cases, n_cyst = apply_cyst_oversample(
            train_cases, repeat=args.cyst_oversample_repeat,
            cache_file=args.cyst_cache, seed=SPLIT_SEED)
        print(f"cyst oversample x{args.cyst_oversample_repeat}: "
              f"{n_cyst} cyst cases, train list {n_before} -> "
              f"{len(train_cases)} entries")

    if args.cyst_centred_sampling:
        print(f"cyst-centred sampling ON: prob "
              f"{args.cyst_centred_prob} per train case, native-"
              f"resolution patch centred on cyst > tumor > kidney; "
              f"val still uses the whole-volume resize path")

    train_ds = KiTS23Dataset(train_cases, args.patch_size, augment=True,
                             centred_sampling=args.cyst_centred_sampling,
                             centred_prob=args.cyst_centred_prob)
    val_ds = KiTS23Dataset(val_cases, args.patch_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    model = LightUNet3D(base_channels=args.base_channels).to(device)
    print(f"params: {count_parameters(model):,} "
          f"({count_parameters(model) / 1e6:.2f}M)")
    loss_fn = KidneyLoss(class_weights=weights,
                         cyst_tversky_weight=args.cyst_tversky_weight,
                         extra_cyst_dice_weight=args.extra_cyst_dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs or (PHASE1_EPOCHS if args.phase == "phase1" else 80))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = args.epochs or (PHASE1_EPOCHS if args.phase == "phase1" else 80)
    start_epoch = 0
    best_fg_dice = -1.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_fg_dice = ckpt.get("best_fg_dice", -1.0)
        print(f"resumed from {args.resume}, epoch {start_epoch}")

    history = []
    # Keep earlier epochs when resuming into the SAME out-dir, so
    # history.json is the full run, not just the resumed tail.
    _hist_file = args.out_dir / "history.json"
    if _hist_file.is_file():
        try:
            history = json.loads(_hist_file.read_text())
            history = [r for r in history
                       if r.get("epoch", 0) < start_epoch + 1]
        except (json.JSONDecodeError, TypeError):
            history = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, loss_fn, optimizer,
                             device, scaler, use_amp)
        va, per_class, fg_dice = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": tr["total"],
               "train_dice_loss": tr["dice"], "val_loss": va["total"],
               "val_fg_dice": fg_dice,
               "dice_bg": per_class[0], "dice_kidney": per_class[1],
               "dice_tumor": per_class[2], "dice_cyst": per_class[3],
               "minutes": (time.time() - t0) / 60}
        history.append(row)
        print(f"epoch {epoch + 1}/{epochs} train_loss={tr['total']:.4f} "
              f"val_fg_dice={fg_dice:.4f} "
              f"[kidney {per_class[1]:.3f} tumor {per_class[2]:.3f} "
              f"cyst {per_class[3]:.3f}] {row['minutes']:.1f} min")
        (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
        is_best = fg_dice > best_fg_dice
        if is_best:
            best_fg_dice = fg_dice
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_fg_dice": best_fg_dice,
                        "base_channels": args.base_channels,
                        "patch_size": list(args.patch_size)},
                       args.out_dir / "best.pt")
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == epochs:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_fg_dice": best_fg_dice,
                        "base_channels": args.base_channels,
                        "patch_size": list(args.patch_size)},
                       args.out_dir / f"checkpoint_epoch{epoch + 1}.pt")
    print(f"done. best val foreground Dice = {best_fg_dice:.4f} "
          f"(measured on {len(val_cases)} val cases — this is the only "
          f"Dice number to quote; do not quote a target as a result).")


if __name__ == "__main__":
    main()
