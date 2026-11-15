"""
KidneyScan AI — Evaluation (KiTS23)
====================================
Metrics, reported per class and split by tumor size and cyst presence.

Metrics (Gaps 1, 2, 4):
  - Dice per class (kidney, tumor, cyst, background): overlap 0-1.
  - Dice by tumor size: tumor <4 cm vs >=4 cm. Tumor size = longest
    diameter of the tumor mask in cm (RECIST-style, longest axis of the
    tumor bounding box in physical mm). Small tumors fail first in the
    literature, so they get their own number.
  - Cyst present vs absent: Dice for tumor/kidney split by whether the
    ground truth has any cyst. Cysts confuse the model into calling
    tumor a cyst and vice versa.
  - Surface Dice (tolerance 1 mm): edge overlap. Punishes bad borders.
  - HD95: 95th-percentile Hausdorff distance in mm. Punishes far-off
    false positives. Both need a distance transform (scipy if present;
    a pure-numpy fallback is used otherwise, slower but works).

Rules:
  - A class absent in BOTH prediction and ground truth scores None
    (not 1.0 and not 0.0) — there is nothing to measure. It is excluded
    from means, and the count of such cases is reported.
  - A class present in ground truth but missing in prediction scores 0.0.
  - NO invented numbers: every value here is computed from real masks.
    If you have not run it, there is no number to quote.

Usage:
    python3 evaluate.py --pred-dir preds/ --gt-dir data/ --out report.json
    (pred dir: one <case>_pred.nii.gz per case; gt dir: case folders
     with segmentation.nii.gz — or pass --gt-pattern.)
    Or import the functions in a notebook / train.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    nib = None
    NIBABEL_AVAILABLE = False

try:
    from scipy import ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    ndi = None
    SCIPY_AVAILABLE = False

NUM_CLASSES = 4
CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
TUMOR_SIZE_CUTOFF_CM = 4.0


# ------------------------------------------------------------- helpers ---

def _binary_dice(pred: np.ndarray, gt: np.ndarray):
    """Dice for one binary mask. None if both are empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return None
    inter = int((pred & gt).sum())
    denom = int(pred.sum() + gt.sum())
    if denom == 0:
        return None
    return 2.0 * inter / denom


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boundary voxels of a binary mask (mask minus its erosion)."""
    if SCIPY_AVAILABLE:
        eroded = ndi.binary_erosion(mask)
        return mask & ~eroded
    # Fallback: a voxel is on the surface if any 6-neighbour is outside.
    surf = np.zeros_like(mask)
    surf[1:, :, :] |= mask[1:, :, :] & ~mask[:-1, :, :]
    surf[:-1, :, :] |= mask[:-1, :, :] & ~mask[1:, :, :]
    surf[:, 1:, :] |= mask[:, 1:, :] & ~mask[:, :-1, :]
    surf[:, :-1, :] |= mask[:, :-1, :] & ~mask[:, 1:, :]
    surf[:, :, 1:] |= mask[:, :, 1:] & ~mask[:, :, :-1]
    surf[:, :, :-1] |= mask[:, :, :-1] & ~mask[:, :, 1:]
    return surf


def _distance_transform(mask: np.ndarray, spacing) -> np.ndarray:
    """Distance (mm) from each voxel to the nearest True voxel."""
    if SCIPY_AVAILABLE:
        return ndi.distance_transform_edt(~mask, sampling=spacing)
    # Pure-numpy fallback: exact only for small volumes; slow for big ones.
    # Good enough for a smoke test, not for a full 512x512 CT.
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.full(mask.shape, np.inf)
    dist = np.full(mask.shape, np.inf)
    it = np.nditer(dist, flags=["multi_index"], op_flags=["readwrite"])
    pts = coords * np.array(spacing)
    for _ in it:
        idx = np.array(it.multi_index) * np.array(spacing)
        dist[it.multi_index] = np.min(np.sqrt(((pts - idx) ** 2).sum(axis=1)))
    return dist


def _surface_dice_one(pred: np.ndarray, gt: np.ndarray, spacing,
                      tolerance_mm: float = 1.0):
    """Surface Dice for one binary mask. None if both are empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return None
    if not pred.any() or not gt.any():
        return 0.0
    pred_surf = _surface(pred)
    gt_surf = _surface(gt)
    dist_to_gt = _distance_transform(gt_surf, spacing)
    dist_to_pred = _distance_transform(pred_surf, spacing)
    pred_ok = (dist_to_gt[pred_surf] <= tolerance_mm).sum()
    gt_ok = (dist_to_pred[gt_surf] <= tolerance_mm).sum()
    denom = int(pred_surf.sum() + gt_surf.sum())
    if denom == 0:
        return None
    return (pred_ok + gt_ok) / denom


def _hd95_one(pred: np.ndarray, gt: np.ndarray, spacing):
    """95th-percentile Hausdorff distance (mm). None if both empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return None
    if not pred.any() or not gt.any():
        return None  # undefined: one side has no surface at all
    pred_surf = _surface(pred)
    gt_surf = _surface(gt)
    d1 = _distance_transform(gt_surf, spacing)[pred_surf]
    d2 = _distance_transform(pred_surf, spacing)[gt_surf]
    both = np.concatenate([d1, d2])
    return float(np.percentile(both, 95))


def _longest_diameter_cm(mask: np.ndarray, spacing) -> float:
    """Longest side of the bounding box, in cm (tumor size proxy)."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0.0
    extents_mm = (coords.max(axis=0) - coords.min(axis=0) + 1) * np.array(spacing)
    return float(extents_mm.max() / 10.0)


# ------------------------------------------------------------- per case ---

def dice_per_class(pred: np.ndarray, gt: np.ndarray):
    """Dice per class. Returns dict {class_index: float or None}."""
    return {c: _binary_dice(pred == c, gt == c) for c in range(NUM_CLASSES)}


def surface_dice_per_class(pred: np.ndarray, gt: np.ndarray, spacing,
                           tolerance_mm: float = 1.0):
    return {c: _surface_dice_one(pred == c, gt == c, spacing, tolerance_mm)
            for c in range(NUM_CLASSES)}


def hd95_per_class(pred: np.ndarray, gt: np.ndarray, spacing):
    return {c: _hd95_one(pred == c, gt == c, spacing)
            for c in range(NUM_CLASSES)}


def evaluate_case(pred: np.ndarray, gt: np.ndarray, spacing) -> dict:
    """All metrics for one case, plus its size/cyst group labels."""
    dice = dice_per_class(pred, gt)
    surf = surface_dice_per_class(pred, gt, spacing)
    hd = hd95_per_class(pred, gt, spacing)
    tumor_size_cm = _longest_diameter_cm(gt == 2, spacing)
    return {
        "dice": dice,
        "surface_dice": surf,
        "hd95_mm": hd,
        "tumor_size_cm": round(tumor_size_cm, 2),
        "tumor_size_group": ("<4cm" if tumor_size_cm < TUMOR_SIZE_CUTOFF_CM
                             else ">=4cm") if tumor_size_cm > 0 else "no tumor",
        "cyst_present": bool((gt == 3).any()),
        "pred_cyst_present": bool((pred == 3).any()),
    }


# -------------------------------------------------------------- summary ---

def _mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def summarize(results: list[dict]) -> dict:
    """Aggregate a list of per-case dicts into the report table."""
    n = len(results)
    out = {"n_cases": n, "per_class": {}, "by_tumor_size": {},
           "by_cyst_present": {}}
    for c in range(1, NUM_CLASSES):
        name = CLASS_NAMES[c]
        out["per_class"][name] = {
            "dice_mean": _mean_or_none([r["dice"][c] for r in results]),
            "dice_n_measured": sum(1 for r in results if r["dice"][c] is not None),
            "surface_dice_mean": _mean_or_none(
                [r["surface_dice"][c] for r in results]),
            "hd95_mean_mm": _mean_or_none([r["hd95_mm"][c] for r in results]),
        }
    for group in ("<4cm", ">=4cm"):
        sub = [r for r in results if r["tumor_size_group"] == group]
        out["by_tumor_size"][group] = {
            "n_cases": len(sub),
            "tumor_dice_mean": _mean_or_none([r["dice"][2] for r in sub]),
            "kidney_dice_mean": _mean_or_none([r["dice"][1] for r in sub]),
        }
    for present in (True, False):
        sub = [r for r in results if r["cyst_present"] is present]
        label = "cyst present" if present else "cyst absent"
        out["by_cyst_present"][label] = {
            "n_cases": len(sub),
            "tumor_dice_mean": _mean_or_none([r["dice"][2] for r in sub]),
            "cyst_dice_mean": _mean_or_none([r["dice"][3] for r in sub]),
            "kidney_dice_mean": _mean_or_none([r["dice"][1] for r in sub]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate KidneyScan predictions")
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="Folder with <case>_pred.nii.gz files")
    ap.add_argument("--gt-dir", type=Path, required=True,
                    help="Folder with case subfolders holding segmentation.nii.gz")
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    args = ap.parse_args()
    if not NIBABEL_AVAILABLE:
        raise SystemExit("nibabel is required: pip install nibabel")

    results = []
    for pred_file in sorted(args.pred_dir.glob("*_pred.nii.gz")):
        case = pred_file.name.replace("_pred.nii.gz", "")
        gt_file = args.gt_dir / case / "segmentation.nii.gz"
        if not gt_file.exists():
            print(f"skip {case}: no ground truth at {gt_file}")
            continue
        pred = np.asanyarray(nib.load(str(pred_file)).dataobj).astype(np.int64)
        gt_nii = nib.load(str(gt_file))
        gt = np.asanyarray(gt_nii.dataobj).astype(np.int64)
        spacing = tuple(float(s) for s in gt_nii.header.get_zooms()[:3])
        if pred.shape != gt.shape:
            print(f"skip {case}: shape mismatch pred {pred.shape} vs gt {gt.shape}")
            continue
        r = evaluate_case(pred, gt, spacing)
        r["case"] = case
        results.append(r)
        d = r["dice"]
        print(f"{case}: tumor_size={r['tumor_size_cm']}cm "
              f"dice kidney={d[1]} tumor={d[2]} cyst={d[3]}")

    report = {"per_case": results, "summary": summarize(results)}
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{len(results)} cases evaluated. Summary:")
    print(json.dumps(report["summary"], indent=2, default=str))
    print(f"Report saved to {args.out}. No numbers here are claimed "
          f"until this script has actually run on real data.")


if __name__ == "__main__":
    main()
