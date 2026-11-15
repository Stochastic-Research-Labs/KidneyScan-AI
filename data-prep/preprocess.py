#!/usr/bin/env python3
"""Preprocess KiTS23 CT volumes for KidneyScan AI.

WHAT WE REUSED (nnU-Net v2, Isensee et al., Nature Methods 2021,
https://github.com/MIC-DKFZ/nnUNet, Apache-2.0) — see README.md reuse table:
  1. CT intensity step: clip to the 0.5 / 99.5 percentiles of the
     foreground, then z-score with the foreground mean and std.
     (nnU-Net's published CT rule.)
  2. Resample all cases to one common voxel spacing before training.
     (nnU-Net uses the dataset median spacing; we use a --target-spacing
     you set after measuring it with check_data.py, default 1 mm isotropic.)
  3. Crop to the region of interest (nnU-Net crops to the non-zero region;
     we crop to the kidney mask bounding box + margin, because the kidney
     is small inside a 512x512 CT).
  4. Output layout: nnU-Net-compatible imagesTr / labelsTr + dataset.json,
     so the same data can be run through real nnU-Net later as a baseline.

WHAT WE CHANGED, and why:
  * We did NOT copy nnU-Net code. These are small functions we wrote
    ourselves that follow the published steps above.
  * Percentiles are computed per case (not from the whole dataset),
    so one case can be preprocessed alone on Kaggle.
  * Fixed HU clip (--clip-min/--clip-max) is an optional override, not
    the default. Default is the percentile rule.
  * Crop uses the ground-truth kidney mask when it exists. For test
    cases with no mask, it falls back to the non-zero image region,
    like nnU-Net does.
  * No patch extraction or augmentation here — that belongs to training.

Usage:
    python3 preprocess.py --data ./raw --out ./preprocessed
    python3 preprocess.py --data ./raw --out ./preprocessed --target-spacing 1 1 2
    python3 preprocess.py --data ./raw --out ./preprocessed --clip-min -100 --clip-max 400

Needs: pip install nibabel numpy scipy
Exit code: 0 = all cases done, 1 = some case failed, 2 = missing library.
"""
import argparse
import json
import os
import sys

LABELS = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}


# ---------------------------------------------------------------- pure numpy
def clip_and_normalize(image, foreground_mask, clip_min=None, clip_max=None):
    """Clip CT values, then z-score with foreground mean/std.

    Default (nnU-Net CT rule): clip to the 0.5 and 99.5 percentiles of
    the foreground voxels. Pass clip_min/clip_max to use fixed HU instead.
    Returns float32 array; background voxels are set to 0 after normalizing.
    """
    import numpy as np

    img = image.astype(np.float32)
    fg = foreground_mask.astype(bool)
    if not fg.any():
        fg = np.ones_like(img, dtype=bool)
    if clip_min is None or clip_max is None:
        lo, hi = np.percentile(img[fg], [0.5, 99.5])
        clip_min = float(lo) if clip_min is None else clip_min
        clip_max = float(hi) if clip_max is None else clip_max
    clipped = np.clip(img, clip_min, clip_max)
    mean = float(clipped[fg].mean())
    std = float(clipped[fg].std())
    if std < 1e-6:
        std = 1.0
    out = (clipped - mean) / std
    out[~fg] = 0.0
    return out.astype(np.float32), {"clip": [clip_min, clip_max],
                                    "mean": mean, "std": std}


def crop_to_roi(image, mask, margin=8):
    """Crop image (and mask) to the bounding box of mask labels > 0.

    Returns (cropped_image, cropped_mask, (z0, y0, x0) start corner).
    If the mask is empty, returns the inputs unchanged with corner (0,0,0).
    """
    import numpy as np

    fg = mask > 0
    if not fg.any():
        return image, mask, (0, 0, 0)
    coords = np.argwhere(fg)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    lo = np.maximum(lo - margin, 0)
    hi = np.minimum(hi + margin, np.array(image.shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return image[sl], mask[sl], tuple(int(v) for v in lo)


def dice_score(pred, truth):
    """Dice overlap, 0 to 1. Empty vs empty = 1.0 (nothing to miss)."""
    import numpy as np

    pred = pred.astype(bool)
    truth = truth.astype(bool)
    inter = int((pred & truth).sum())
    total = int(pred.sum()) + int(truth.sum())
    if total == 0:
        return 1.0
    return 2.0 * inter / total


# ---------------------------------------------------------------- NIfTI I/O
def resample_volume(data, old_spacing, new_spacing, order):
    """Resample a 3D array to new_spacing (mm) with scipy zoom."""
    from scipy.ndimage import zoom

    import numpy as np

    old = np.array(old_spacing, dtype=float)
    new = np.array(new_spacing, dtype=float)
    factors = old / new
    return zoom(data, factors, order=order)


def _shift_affine_for_crop(affine, corner):
    import numpy as np

    new = np.array(affine, dtype=float)
    new[:3, 3] = np.array(affine[:3, 3]) + np.array(affine[:3, :3]) @ np.array(corner)
    return new


def _scale_affine_for_spacing(affine, old_spacing, new_spacing):
    import numpy as np

    new = np.array(affine, dtype=float)
    scale = np.array(new_spacing, dtype=float) / np.array(old_spacing, dtype=float)
    for i in range(3):  # column i is direction * spacing i
        new[:3, i] = np.array(affine[:3, i]) * scale[i]
    return new


def preprocess_case(img_path, msk_path, out_img, out_msk, target_spacing,
                    clip_min, clip_max, margin):
    """Full pipeline for one case. Returns a per-case report dict."""
    import nibabel as nib
    import numpy as np

    img = nib.load(img_path)
    data = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    affine = np.array(img.affine)
    mask = None
    if msk_path and os.path.isfile(msk_path):
        mask = np.asanyarray(nib.load(msk_path).dataobj)
        if mask.shape != data.shape:
            raise ValueError(f"mask shape {mask.shape} != image {data.shape}")
    report = {"case": os.path.basename(img_path),
              "input_shape": [int(s) for s in data.shape],
              "input_spacing_mm": spacing}

    # 1. Resample to common spacing (image: linear, mask: nearest).
    if [round(s, 3) for s in spacing] != [round(s, 3) for s in target_spacing]:
        data = resample_volume(data, spacing, target_spacing, order=1)
        affine = _scale_affine_for_spacing(affine, spacing, target_spacing)
        if mask is not None:
            mask = resample_volume(mask, spacing, target_spacing, order=0)
            mask = np.rint(mask).astype(mask.dtype)
        spacing = list(target_spacing)
    report["resampled_shape"] = [int(s) for s in data.shape]

    # 2. Foreground = kidney region if we have a mask, else non-zero image.
    fg = (mask > 0) if mask is not None else (data != 0)
    # 3. Clip + z-score (nnU-Net CT rule).
    data, stats = clip_and_normalize(data, fg, clip_min, clip_max)
    report.update({"clip_hu": stats["clip"], "foreground_mean": round(stats["mean"], 3),
                   "foreground_std": round(stats["std"], 3)})

    # 4. Crop to kidney ROI (+ margin). Test cases without a mask keep
    #    the full volume (crop_to_roi returns inputs unchanged).
    corner = (0, 0, 0)
    if mask is not None:
        data, mask, corner = crop_to_roi(data, mask, margin=margin)
        affine = _shift_affine_for_crop(affine, corner)
    report["crop_corner"] = list(corner)
    report["output_shape"] = [int(s) for s in data.shape]

    nib.save(nib.Nifti1Image(data, affine), out_img)
    if mask is not None:
        nib.save(nib.Nifti1Image(mask.astype(np.int16), affine), out_msk)
    report["output_image"] = out_img
    report["output_mask"] = out_msk if mask is not None else None
    return report


def main(argv=None):
    from check_data import find_pairs  # same folder: case discovery

    ap = argparse.ArgumentParser(description="Preprocess KiTS23 CT volumes.")
    ap.add_argument("--data", required=True, help="Raw data folder (see check_data.py).")
    ap.add_argument("--out", required=True, help="Output folder (nnU-Net layout).")
    ap.add_argument("--target-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="Voxel spacing in mm after resampling. Default: 1 1 1.")
    ap.add_argument("--clip-min", type=float, default=None,
                    help="Fixed HU clip low end (default: 0.5 percentile).")
    ap.add_argument("--clip-max", type=float, default=None,
                    help="Fixed HU clip high end (default: 99.5 percentile).")
    ap.add_argument("--margin", type=int, default=8,
                    help="Voxels of margin around the kidney crop. Default: 8.")
    args = ap.parse_args(argv)

    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print("ERROR: nibabel, numpy and scipy are needed.", file=sys.stderr)
        print("Run: pip install nibabel numpy scipy", file=sys.stderr)
        return 2

    pairs = find_pairs(args.data)
    if not pairs:
        print(f"ERROR: no cases found in {args.data}", file=sys.stderr)
        return 1
    img_dir = os.path.join(args.out, "imagesTr")
    lbl_dir = os.path.join(args.out, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    reports, failed = [], []
    for i, case_id in enumerate(sorted(pairs)):
        img, msk = pairs[case_id]
        name = f"case_{i:04d}"
        out_img = os.path.join(img_dir, name + "_0000.nii.gz")
        out_msk = os.path.join(lbl_dir, name + ".nii.gz")
        try:
            rep = preprocess_case(img, msk, out_img, out_msk,
                                  args.target_spacing, args.clip_min,
                                  args.clip_max, args.margin)
            rep["case"] = case_id
            reports.append(rep)
            print(f"  [{len(reports)}/{len(pairs)}] {case_id} "
                  f"{rep['input_shape']} -> {rep['output_shape']}")
        except Exception as exc:
            failed.append({"case": case_id, "error": str(exc)})
            print(f"  FAILED {case_id}: {exc}", file=sys.stderr)

    dataset_json = {
        "name": "KiTS23 KidneyScan",
        "description": "KiTS23 kidney/tumor/cyst CT, preprocessed for KidneyScan AI",
        "reference": "https://kits-challenge.org/kits23/",
        "licence": "CC BY-NC-SA 4.0",
        "modality": {"0": "CT"},
        "labels": {"0": "background", "1": "kidney", "2": "tumor", "3": "cyst"},
        "numTraining": len(reports),
        "file_ending": ".nii.gz",
        "training": [{"image": f"./imagesTr/{os.path.basename(r['output_image'])}",
                      "label": f"./labelsTr/{os.path.basename(r['output_mask'])}"}
                     for r in reports if r["output_mask"]],
    }
    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)
    with open(os.path.join(args.out, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump({"target_spacing_mm": args.target_spacing,
                   "cases": reports, "failed": failed}, f, indent=2)
    print(f"Done. {len(reports)} cases OK, {len(failed)} failed. Output: {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
