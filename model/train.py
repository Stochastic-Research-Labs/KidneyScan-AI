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
    """

    def __init__(self, cases, patch_size=(128, 128, 64), augment=False):
        self.cases = [(str(cid), Path(img), Path(msk)) for cid, img, msk in cases]
        self.patch_size = tuple(patch_size)
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        _case_id, img_path, mask_path = self.cases[idx]
        ct, _spacing = load_nifti(img_path)
        mask, _ = load_nifti(mask_path)
        ct = preprocess_ct(ct)
        ct, mask = crop_to_mask(ct, mask)
        ct = resize_to_patch(ct, self.patch_size, order=1)
        mask = resize_to_patch(mask.astype(np.float32), self.patch_size,
                               order=0).round().astype(np.int64)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        if self.augment:
            ct, mask = augment_pair(ct, mask)
        x = torch.from_numpy(ct[None].astype(np.float32))
        y = torch.from_numpy(mask.astype(np.int64))
        return x, y


def augment_pair(ct: np.ndarray, mask: np.ndarray):
    """Flip / rotate / zoom / noise, applied to image and mask together."""
    rng = np.random
    for axis in range(3):
        if rng.rand() < 0.5:
            ct = np.flip(ct, axis=axis)
            mask = np.flip(mask, axis=axis)
    k = rng.randint(0, 4)  # 0, 90, 180, 270 degrees in the H-W plane
    if k:
        ct = np.rot90(ct, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(1, 2))
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
    # Counts voxels on a small sample (first 20 train cases) to keep it fast.
    try:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _case_id, _img_path, mask_path in train_cases[:20]:
            mask, _ = load_nifti(mask_path)
            for c in range(NUM_CLASSES):
                counts[c] += int((mask == c).sum())
        weights = compute_class_weights(counts).tolist()
        print(f"class weights from real data (bg, kidney, tumor, cyst): {weights}")
    except Exception as e:  # noqa: BLE001 - fall back, do not crash training
        print(f"could not count voxels ({e}); using fallback weights")
        weights = DEFAULT_CLASS_WEIGHTS

    train_ds = KiTS23Dataset(train_cases, args.patch_size, augment=True)
    val_ds = KiTS23Dataset(val_cases, args.patch_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)

    model = LightUNet3D(base_channels=args.base_channels).to(device)
    print(f"params: {count_parameters(model):,} "
          f"({count_parameters(model) / 1e6:.2f}M)")
    loss_fn = KidneyLoss(class_weights=weights)
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
