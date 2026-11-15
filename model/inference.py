"""
KidneyScan AI — Inference (KiTS23)
===================================
Run the trained light 3D U-Net on ONE CT (.nii.gz) and return masks +
volumes + scores, in exactly the JSON shape the web demo expects.

Output contract: see ../../API_CONTRACT.md (canonical v1.1; this
folder's API_CONTRACT.md is only a pointer to it). The JSON returned by
``predict()`` matches the core fields of that contract exactly,
so the web worker can build against it without guessing.

What it does:
  1. Load CT (.nii.gz), clip HU to [-200, 300], normalise to [0, 1].
  2. Run the model on overlapping patches (sliding window) OR on the
     whole volume resized to the training patch size (default: sliding
     window, better for large CTs; --fast uses resize).
  3. Argmax over 4 classes -> mask (0 background, 1 kidney, 2 tumor, 3 cyst).
  4. Volumes in cm3 = voxel count x voxel volume (from NIfTI spacing).
  5. If a ground-truth mask is given (--ground-truth), also report
     per-class Dice, Surface Dice and HD95 (via evaluate.py).

REAL AI ONLY: the output mask comes only from the model. If the model
finds no tumor/cyst, the output says so — nothing is hard-coded or
highlighted by hand. A normal case returns empty tumor/cyst masks and
``"finding": "no tumor or cyst detected"``.

Assistive only: every output carries the warning that a doctor must
confirm it. This is not a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nib = None
    NIBABEL_AVAILABLE = False

try:
    from light_unet3d import LightUNet3D, NUM_CLASSES, CLASS_NAMES
    from train import preprocess_ct
    from evaluate import dice_per_class, surface_dice_per_class, hd95_per_class
except ImportError:  # torch missing — predict() reports this cleanly
    LightUNet3D = None
    NUM_CLASSES = 4
    CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
    preprocess_ct = None
    dice_per_class = surface_dice_per_class = hd95_per_class = None

ASSISTIVE_WARNING = (
    "Assistive tool only. This is not a diagnosis. "
    "A doctor must confirm the result."
)


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load a checkpoint saved by train.py. Returns (model, meta)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LightUNet3D(base_channels=ckpt.get("base_channels", 48))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def sliding_window_predict(model, ct: np.ndarray, patch_size, device: str,
                           overlap: float = 0.5) -> np.ndarray:
    """Run the model on overlapping patches, average the logits.

    ct: normalised 3D array (D, H, W). Returns logits (4, D, H, W).
    """
    patch = tuple(patch_size)
    step = tuple(max(1, int(p * (1 - overlap))) for p in patch)
    logits_sum = np.zeros((NUM_CLASSES,) + ct.shape, dtype=np.float32)
    counts = np.zeros(ct.shape, dtype=np.float32)

    starts = []
    for dim, p, s in zip(ct.shape, patch, step):
        pos = list(range(0, max(1, dim - p + 1), s))
        if pos[-1] + p < dim:
            pos.append(dim - p)
        starts.append(pos)

    with torch.no_grad():
        for d0 in starts[0]:
            for h0 in starts[1]:
                for w0 in starts[2]:
                    patch_ct = ct[d0:d0 + patch[0], h0:h0 + patch[1],
                                  w0:w0 + patch[2]]
                    pad = [(0, p - s) for p, s in zip(patch, patch_ct.shape)]
                    if any(b > 0 for _, b in pad):
                        patch_ct = np.pad(patch_ct, pad)
                    x = torch.from_numpy(patch_ct[None, None]).float().to(device)
                    out = model(x)[0].cpu().numpy()
                    out = out[:, :ct.shape[0] - d0, :ct.shape[1] - h0,
                              :ct.shape[2] - w0]
                    sl = (slice(None), slice(d0, d0 + out.shape[1]),
                          slice(h0, h0 + out.shape[2]),
                          slice(w0, w0 + out.shape[3]))
                    logits_sum[sl] += out
                    counts[d0:d0 + out.shape[1], h0:h0 + out.shape[2],
                           w0:w0 + out.shape[3]] += 1
    counts = np.maximum(counts, 1)
    return logits_sum / counts[None]


def predict(ct_path: Path, checkpoint_path: Path, device: str = "cpu",
            patch_size=(128, 128, 64), fast: bool = False,
            ground_truth_path: Path | None = None) -> dict:
    """Predict on one CT. Returns the API-contract JSON dict (no file I/O)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install torch")
    if not NIBABEL_AVAILABLE:
        raise ImportError("nibabel is required: pip install nibabel")

    t0 = time.time()
    case_id = (ct_path.name[:-len(".nii.gz")] if ct_path.name.endswith(".nii.gz")
               else ct_path.stem) or "case"
    nii = nib.load(str(ct_path))
    ct = np.asanyarray(nii.dataobj).astype(np.float32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])  # mm
    affine = nii.affine
    original_shape = list(ct.shape)
    ct = preprocess_ct(ct)

    model, meta = load_checkpoint(checkpoint_path, device)
    train_patch = tuple(meta.get("patch_size", list(patch_size)))

    if fast:
        # Resize whole volume to the training patch, single forward pass.
        from train import resize_to_patch
        small = resize_to_patch(ct, train_patch, order=1)
        with torch.no_grad():
            logits = model(torch.from_numpy(small[None, None]).float()
                           .to(device))[0].cpu().numpy()
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        mask = np.argmax(probs, axis=0).astype(np.uint8)
        mask = resize_to_patch(mask.astype(np.float32), original_shape,
                               order=0).round().astype(np.uint8)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
    else:
        logits = sliding_window_predict(model, ct, train_patch, device)
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        mask = np.argmax(probs, axis=0).astype(np.uint8)

    voxel_volume_mm3 = float(np.prod(spacing))
    volumes_cm3 = {}
    voxel_counts = {}
    for c in range(1, NUM_CLASSES):
        n = int((mask == c).sum())
        voxel_counts[c] = n
        volumes_cm3[CLASS_NAMES[c]] = round(n * voxel_volume_mm3 / 1000.0, 2)

    tumor_voxels = voxel_counts[2]
    cyst_voxels = voxel_counts[3]
    if tumor_voxels == 0 and cyst_voxels == 0:
        finding = "no tumor or cyst detected"
    elif tumor_voxels > 0 and cyst_voxels == 0:
        finding = "tumor detected, no cyst detected"
    elif tumor_voxels == 0 and cyst_voxels > 0:
        finding = "cyst detected, no tumor detected"
    else:
        finding = "tumor and cyst detected"
    # Canonical contract v1.1: findings object (v1 counts one bounding
    # lesion per class, so cyst_count is 0 or 1 — not connected blobs).
    findings = {
        "tumor_present": tumor_voxels > 0,
        "cyst_present": cyst_voxels > 0,
        "cyst_count": 1 if cyst_voxels > 0 else 0,
        "tumor_volume_cm3": volumes_cm3["tumor"],
        "cyst_volume_cm3": volumes_cm3["cyst"],
    }

    result = {
        "status": "ok",
        "case_id": case_id,
        "mode": "real",
        "is_dummy": False,
        "model": {
            "name": "KidneyScan light 3D U-Net",
            "checkpoint": str(checkpoint_path),
            "base_channels": meta.get("base_channels", 48),
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
        },
        "note": "",
        "input": {
            "file": str(ct_path),
            "shape": original_shape,
            "spacing_mm": [round(s, 3) for s in spacing],
        },
        "shape": original_shape,
        "spacing_mm": [round(s, 3) for s in spacing],
        "finding": finding,
        "findings": findings,
        "volumes_cm3": {
            "kidney": volumes_cm3["kidney"],
            "tumor": volumes_cm3["tumor"],
            "cyst": volumes_cm3["cyst"],
        },
        "voxel_counts": {CLASS_NAMES[c]: voxel_counts[c]
                         for c in range(1, NUM_CLASSES)},
        "mask": {
            "encoding": "nifti-gz file, same shape/orientation as input; "
                        "values 0 background, 1 kidney, 2 tumor, 3 cyst",
            "file": None,  # filled in by main() when --save-mask is used
            "shape": original_shape,
        },
        "scores": None,  # filled in below only when ground truth is given
        "inference_seconds": round(time.time() - t0, 2),
        "warning": ASSISTIVE_WARNING,
    }

    if ground_truth_path is not None:
        gt_nii = nib.load(str(ground_truth_path))
        gt = np.asanyarray(gt_nii.dataobj).astype(np.int64)
        gt = np.clip(gt, 0, NUM_CLASSES - 1)
        if gt.shape != mask.shape:
            raise ValueError(f"ground truth shape {gt.shape} != "
                             f"prediction shape {mask.shape}")
        dice = dice_per_class(mask, gt)
        surf = surface_dice_per_class(mask, gt, spacing)
        hd = hd95_per_class(mask, gt, spacing)
        # Canonical contract v1.1: scores per class WITHOUT background
        # (kidney / tumor / cyst only).
        result["scores"] = {
            "dice": {CLASS_NAMES[c]: (None if dice[c] is None
                                      else round(float(dice[c]), 4))
                     for c in range(1, NUM_CLASSES)},
            "surface_dice_mm1": {CLASS_NAMES[c]: (None if surf[c] is None
                                                 else round(float(surf[c]), 4))
                                for c in range(1, NUM_CLASSES)},
            "hd95_mm": {CLASS_NAMES[c]: (None if hd[c] is None
                                         else round(float(hd[c]), 2))
                        for c in range(1, NUM_CLASSES)},
            "note": "Dice is None for a class absent in both prediction "
                    "and ground truth (no overlap to measure).",
        }
    return result, mask, affine


def main():
    ap = argparse.ArgumentParser(description="KidneyScan inference on one CT")
    ap.add_argument("ct", type=Path, help="Input CT .nii.gz")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, default=None)
    ap.add_argument("--save-mask", type=Path, default=None,
                    help="Where to save the predicted mask .nii.gz")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="Resize whole volume instead of sliding window")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    result, mask, affine = predict(
        args.ct, args.checkpoint, device=args.device, fast=args.fast,
        ground_truth_path=args.ground_truth)
    if args.save_mask is not None:
        nib.save(nib.Nifti1Image(mask, affine), str(args.save_mask))
        result["mask"]["file"] = str(args.save_mask)
    text = json.dumps(result, indent=2)
    if args.out_json is not None:
        args.out_json.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
