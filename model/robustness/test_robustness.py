"""KidneyScan AI — robustness test runner.

Run a model on clean vs corrupted CT and score the fall.
Clean vs noise vs low-res vs missing slices vs intensity shift.

Metrics, per class (kidney=1, tumor=2, cyst=3):
  * Dice            — overlap, 0 to 1, 1 is perfect.
  * Surface Dice    — edge overlap within a tolerance (default 1 mm).
                      Punishes bad borders more than Dice (Gap 4).
  * HD95            — 95th percentile surface distance in mm.
                      Lower is better. Punishes bad edges too.
  * Dice drop       — clean Dice minus corrupted Dice.
                      Bigger drop = less robust. This is the main score.

Gap scores (from the plan / literature review):
  * Cyst: Dice for class 3 alone (Scalco 2024: cyst Dice 0.21).
  * Small tumors: tumor Dice for cases with tumor < 4 cm, apart
    from tumors >= 4 cm (Gap 2).
  * Edges: Surface Dice + HD95, not Dice only (Gap 4).

How to use:
    from test_robustness import run_robustness
    results = run_robustness(predict_fn, volume, ground_truth,
                             spacing_mm=(1.0, 1.0, 1.0))
    # predict_fn(volume) -> 3D label mask, same shape, values 0-3.

No invented numbers: this file computes scores from real inputs
only. If you run it with the dummy predictor below, the scores are
a code smoke test only — do NOT quote them as model results.

The dummy predictor is NOT a model. It returns the ground truth
eroded by 1 voxel, so the metric code has something to score.
Real use: pass your trained model's predict function instead.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from robustness import corrupt

CLASSES = {1: "kidney", 2: "tumor", 3: "cyst"}
CONDITIONS = ["clean", "noise", "lowres", "dropslices", "shift"]
SMALL_TUMOR_CM = 4.0  # plan threshold: small tumor < 4 cm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred, truth):
    """Dice for one binary mask. Both empty -> 1.0 (nothing to miss)."""
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    inter = int(np.logical_and(pred, truth).sum())
    total = int(pred.sum()) + int(truth.sum())
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def _surface(mask):
    """Surface voxels: mask minus its 1-voxel erosion."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded


def surface_dice(pred, truth, spacing_mm=(1.0, 1.0, 1.0), tolerance_mm=1.0):
    """Surface Dice: fraction of surface voxels within tolerance.

    Both empty -> 1.0. One empty, other not -> 0.0.
    Distance is in mm, using the voxel spacing.
    Idea reused from KiTS23 scoring (surface Dice 0.723, winner) and
    nnU-Net evaluation; rewritten here with SciPy distance transforms.
    """
    pred_s = _surface(pred)
    truth_s = _surface(truth)
    if not pred_s.any() and not truth_s.any():
        return 1.0
    if not pred_s.any() or not truth_s.any():
        return 0.0
    # Distance from each pred surface voxel to the nearest truth surface voxel.
    dt_truth = ndimage.distance_transform_edt(~truth_s, sampling=spacing_mm)
    dt_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing_mm)
    ok_pred = float((dt_truth[pred_s] <= tolerance_mm).sum())
    ok_truth = float((dt_pred[truth_s] <= tolerance_mm).sum())
    return (ok_pred + ok_truth) / (float(pred_s.sum()) + float(truth_s.sum()))


def hd95(pred, truth, spacing_mm=(1.0, 1.0, 1.0)):
    """95th percentile Hausdorff distance in mm (lower is better).

    Both empty -> 0.0. One empty, other not -> NaN (no surface to
    compare); callers must show it as "—", not as 0.
    """
    pred_s = _surface(pred)
    truth_s = _surface(truth)
    if not pred_s.any() and not truth_s.any():
        return 0.0
    if not pred_s.any() or not truth_s.any():
        return float("nan")
    dt_truth = ndimage.distance_transform_edt(~truth_s, sampling=spacing_mm)
    dt_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing_mm)
    d1 = dt_truth[pred_s]
    d2 = dt_pred[truth_s]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def per_class_metrics(pred, truth, spacing_mm=(1.0, 1.0, 1.0)):
    """Dice / Surface Dice / HD95 for kidney, tumor, cyst."""
    out = {}
    for label, name in CLASSES.items():
        p = np.asarray(pred) == label
        t = np.asarray(truth) == label
        out[name] = {
            "dice": dice_score(p, t),
            "surface_dice": surface_dice(p, t, spacing_mm),
            "hd95_mm": hd95(p, t, spacing_mm),
        }
    return out


def tumor_size_cm(truth, spacing_mm=(1.0, 1.0, 1.0)):
    """Tumor size as equivalent sphere diameter in cm.

    From tumor voxel count x voxel volume. A sphere with the same
    volume has diameter 2 * (3V / 4pi) ^ (1/3). This is an estimate
    for the <4 cm split only, not a clinical measurement.
    Returns None if there is no tumor in the ground truth.
    """
    n = int((np.asarray(truth) == 2).sum())
    if n == 0:
        return None
    voxel_mm3 = float(np.prod(spacing_mm))
    volume_mm3 = n * voxel_mm3
    diameter_mm = 2.0 * ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return diameter_mm / 10.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def evaluate_one(predict_fn, volume, truth, spacing_mm=(1.0, 1.0, 1.0), severity="moderate"):
    """Score one case under every condition. Returns a dict:

    {condition: {class: {dice, surface_dice, hd95_mm}, ...}, ...}
    plus "tumor_size_cm" and "is_small_tumor" (tumor < 4 cm).
    """
    truth = np.asarray(truth)
    results = {}
    for condition in CONDITIONS:
        if condition == "clean":
            vol = np.asarray(volume)
        else:
            vol = corrupt(volume, condition, severity=severity)
        pred = np.asarray(predict_fn(vol))
        if pred.shape != truth.shape:
            raise ValueError(f"Prediction shape {pred.shape} != ground truth {truth.shape}.")
        results[condition] = per_class_metrics(pred, truth, spacing_mm)
    size = tumor_size_cm(truth, spacing_mm)
    results["tumor_size_cm"] = size
    results["is_small_tumor"] = (size is not None and size < SMALL_TUMOR_CM)
    return results


def dice_drop(results, corrupted, class_name):
    """Dice drop for one class: clean Dice minus corrupted Dice."""
    return results["clean"][class_name]["dice"] - results[corrupted][class_name]["dice"]


def summarize_drops(results):
    """Dice drop table: {corrupted condition: {class: drop}}."""
    return {
        cond: {name: dice_drop(results, cond, name) for name in CLASSES.values()}
        for cond in CONDITIONS
        if cond != "clean"
    }


def run_robustness(predict_fn, volume, truth, spacing_mm=(1.0, 1.0, 1.0), severity="moderate"):
    """Full run for one case. Same as evaluate_one, plus the drop table.

    Returns {"per_condition": ..., "dice_drop": ..., "tumor_size_cm": ...,
             "is_small_tumor": ...}. Print it, or fill the report
    template in docs/robustness-report-template.md with it.
    """
    results = evaluate_one(predict_fn, volume, truth, spacing_mm, severity)
    return {
        "per_condition": {c: results[c] for c in CONDITIONS},
        "dice_drop": summarize_drops(results),
        "tumor_size_cm": results["tumor_size_cm"],
        "is_small_tumor": results["is_small_tumor"],
    }


# ---------------------------------------------------------------------------
# Dummy predictor + synthetic data — smoke test only, NOT a model
# ---------------------------------------------------------------------------

def dummy_predict(volume):
    """NOT a model. For pipeline testing only.

    Needs the ground truth, so it cannot run on a real new case.
    It is only used by the __main__ smoke test below, to prove the
    metric code runs. Never quote its scores as model results.
    """
    raise NotImplementedError("dummy_predict needs ground truth; use make_dummy_predictor.")


def make_dummy_predictor(truth, erode=True):
    """Return a fake predict_fn that returns the truth, eroded by
    1 voxel if erode=True. Scores from it are meaningless as model
    results — they only check that this file runs end to end."""
    truth = np.asarray(truth)

    def predict(volume):
        out = truth.copy()
        if erode:
            for label in (1, 2, 3):
                mask = truth == label
                if mask.any():
                    out[mask & ~ndimage.binary_erosion(mask)] = 0
        return out

    return predict


def make_synthetic_case(shape=(16, 32, 32), seed=0):
    """Small fake CT + mask for the smoke test. Kidney = ball,
    tumor = small ball inside it, cyst = tiny ball. HU values are
    made up for testing only, not real CT values."""
    rng = np.random.default_rng(seed)
    z, y, x = np.indices(shape)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    vol = rng.normal(0, 5, size=shape).astype(np.float32)  # background
    truth = np.zeros(shape, dtype=np.int32)
    kidney = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= 10 ** 2
    tumor = (z - cz) ** 2 + (y - (cy - 3)) ** 2 + (x - (cx + 3)) ** 2 <= 3 ** 2
    cyst = (z - (cz + 4)) ** 2 + (y - (cy + 4)) ** 2 + (x - (cx - 4)) ** 2 <= 2 ** 2
    vol[kidney] = 30.0
    vol[tumor] = 60.0
    vol[cyst] = 10.0
    truth[kidney] = 1
    truth[tumor] = 2
    truth[cyst] = 3
    return vol, truth


if __name__ == "__main__":
    # Smoke test: does the code run? Scores here are NOT results.
    vol, truth = make_synthetic_case()
    out = run_robustness(make_dummy_predictor(truth), vol, truth,
                         spacing_mm=(1.0, 1.0, 1.0), severity="moderate")
    print("Smoke test only — these numbers are NOT model results.")
    print(f"Tumor size (fake case): {out['tumor_size_cm']:.2f} cm, "
          f"small tumor (<4 cm): {out['is_small_tumor']}")
    for cond, scores in out["per_condition"].items():
        dice = {k: round(v["dice"], 3) for k, v in scores.items()}
        print(f"  {cond:12s} Dice {dice}")
    print("  Dice drop (clean - corrupted):")
    for cond, drops in out["dice_drop"].items():
        print(f"    {cond:12s} " + ", ".join(f"{k} {v:+.3f}" for k, v in drops.items()))
    print("OK — test_robustness.py runs.")
