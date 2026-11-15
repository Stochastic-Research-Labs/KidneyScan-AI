#!/usr/bin/env python3
"""Test all data-prep logic on a tiny synthetic volume.

No real data needed. No download. No GPU.
Builds a small fake CT (32x32x16) with a kidney blob, a small tumor
blob and a tiny cyst blob, then checks every function the real pipeline
uses. Prints PASS/FAIL per test and exits 1 if any test fails.

Tests:
  1. clip_and_normalize — output is finite, foreground mean ~0, std ~1.
  2. crop_to_roi — crop contains the kidney and is smaller than the input.
  3. dice_score — perfect match = 1.0, no overlap = 0.0, empty = 1.0.
  4. per_class_dice — kidney/tumor/cyst Dice are all 1.0 on identical masks.
  5. tumor_size_group — small vs large vs none, at the 4 cm cutoff.
  6. make_split — 400/89 ratio on fake stats, no train/val overlap,
     fixed seed gives the same split twice, cyst cases appear in both splits.
  7. stratified_report — groups Dice by size group and cyst presence.
  8. check_data + preprocess end-to-end on synthetic .nii.gz files
     (runs only if nibabel is installed; otherwise SKIP, not FAIL).
     Needs scipy too for the resample step (SKIP without it).

Usage:
    python3 test_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import clip_and_normalize, crop_to_roi, dice_score  # noqa: E402
from splits import (make_split, per_class_dice, stratified_report,  # noqa: E402
                    tumor_size_group)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))


def make_synthetic(shape=(32, 32, 16)):
    """Fake CT + mask. Kidney: big blob. Tumor: small blob. Cyst: tiny blob."""
    rng = np.random.default_rng(0)
    image = rng.normal(50, 30, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.int16)
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    kidney = ((zz - 16) ** 2 + (yy - 16) ** 2) / 49.0 + ((xx - 8) ** 2) / 16.0 <= 1
    tumor = ((zz - 18) ** 2 + (yy - 18) ** 2) / 4.0 + ((xx - 8) ** 2) / 4.0 <= 1
    cyst = ((zz - 13) ** 2 + (yy - 13) ** 2) / 1.0 + ((xx - 7) ** 2) / 1.0 <= 1
    mask[kidney] = 1
    mask[tumor] = 2
    mask[cyst] = 3
    image[kidney] = 120.0
    image[tumor] = 60.0
    image[cyst] = 10.0
    return image, mask


def fake_stats(n=489):
    """Fake check_data.py output: half small tumors, half large, half cyst."""
    cases = []
    for i in range(n):
        small = (i % 2 == 0)
        cyst = (i % 2 == 0) if small else (i % 3 == 0)
        tumor_vox = 8000 if small else 60000  # ~8 / ~60 cm3 at 1 mm voxels
        cases.append({
            "case": f"case_{i:05d}",
            "errors": [],
            "label_counts": {"kidney": 50000, "tumor": tumor_vox,
                             "cyst": 500 if cyst else 0},
            "volumes_cm3": {"kidney": 50.0, "tumor": float(tumor_vox) / 1000.0,
                            "cyst": 0.5 if cyst else 0.0},
            "voxel_spacing_mm": [1.0, 1.0, 1.0],
        })
    return cases


def main():
    image, mask = make_synthetic()

    # 1. clip + normalize
    norm, stats = clip_and_normalize(image, mask > 0)
    fg = norm[mask > 0]
    check("clip_and_normalize finite", np.isfinite(norm).all())
    check("clip_and_normalize mean~0 std~1",
          abs(float(fg.mean())) < 0.05 and 0.9 < float(fg.std()) < 1.1,
          f"mean={float(fg.mean()):.3f} std={float(fg.std()):.3f}")

    # 2. crop to kidney ROI
    c_img, c_msk, corner = crop_to_roi(image, mask, margin=2)
    check("crop_to_roi smaller", c_img.shape < image.shape,
          f"{image.shape} -> {c_img.shape}")
    check("crop_to_roi keeps kidney", int((c_msk == 1).sum()) == int((mask == 1).sum()))
    check("crop_to_roi keeps tumor+cyst",
          int((c_msk == 2).sum()) == int((mask == 2).sum())
          and int((c_msk == 3).sum()) == int((mask == 3).sum()))

    # 3. Dice
    check("dice perfect = 1.0", dice_score(mask == 2, mask == 2) == 1.0)
    check("dice no overlap = 0.0",
          dice_score(mask == 2, mask == 3) == 0.0)
    check("dice empty vs empty = 1.0",
          dice_score(np.zeros(5, bool), np.zeros(5, bool)) == 1.0)

    # 4. per-class Dice
    d = per_class_dice(mask, mask)
    check("per_class_dice all 1.0", all(v == 1.0 for v in d.values()), str(d))

    # 5. tumor size groups (equivalent-sphere diameter)
    check("tumor_size_group small", tumor_size_group(8.0) == "small")
    check("tumor_size_group large", tumor_size_group(60.0) == "large")
    check("tumor_size_group none", tumor_size_group(0.0) == "none")

    # 6. stratified split on 489 fake cases
    cases = fake_stats(489)
    train, val, summary = make_split(cases)
    check("split sizes 400/89", len(train) == 400 and len(val) == 89,
          f"{len(train)}/{len(val)}")
    check("split no overlap", not (set(train) & set(val)))
    train2, val2, _ = make_split(cases)
    check("split seed is fixed", train == train2 and val == val2)
    cyst_train = sum(1 for c in cases if c["case"] in set(train)
                     and c["label_counts"]["cyst"] > 0)
    cyst_val = sum(1 for c in cases if c["case"] in set(val)
                   and c["label_counts"]["cyst"] > 0)
    check("split cyst in both parts", cyst_train > 0 and cyst_val > 0,
          f"cyst train={cyst_train} val={cyst_val}")

    # 7. stratified report
    rows = [{"case": "a", "tumor_size_group": "small", "cyst_present": True,
             "dice": {1: 0.9, 2: 0.8, 3: 0.5}},
            {"case": "b", "tumor_size_group": "large", "cyst_present": False,
             "dice": {1: 0.95, 2: 0.85, 3: 1.0}}]
    rep = stratified_report(rows)
    check("stratified_report groups",
          rep["small__cyst"]["dice_label_3"] == 0.5
          and rep["large__no_cyst"]["dice_label_2"] == 0.85, str(rep))

    # 8. end-to-end on synthetic .nii.gz files (needs nibabel + scipy)
    try:
        import nibabel as nib  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print("SKIP  end-to-end nifti test — nibabel/scipy not installed here")
        print("      (install on Kaggle with: pip install nibabel scipy)")
    else:
        import tempfile

        from check_data import check_case, find_pairs
        from preprocess import preprocess_case

        with tempfile.TemporaryDirectory() as tmp:
            case_dir = os.path.join(tmp, "case_00000")
            os.makedirs(case_dir)
            nib.save(nib.Nifti1Image(image, np.eye(4)),
                     os.path.join(case_dir, "imaging.nii.gz"))
            nib.save(nib.Nifti1Image(mask, np.eye(4)),
                     os.path.join(case_dir, "segmentation.nii.gz"))
            pairs = find_pairs(tmp)
            check("find_pairs finds synthetic case", len(pairs) == 1, str(pairs))
            res = check_case("case_00000", pairs["case_00000"][0],
                             pairs["case_00000"][1])
            check("check_case no errors", res["errors"] == [], str(res["errors"]))
            check("check_case label counts",
                  res["label_counts"].get("cyst", 0) == int((mask == 3).sum()),
                  str(res["label_counts"]))
            out_img = os.path.join(tmp, "out_0000.nii.gz")
            out_msk = os.path.join(tmp, "out.nii.gz")
            rep2 = preprocess_case(pairs["case_00000"][0], pairs["case_00000"][1],
                                   out_img, out_msk, [1.0, 1.0, 1.0],
                                   None, None, margin=2)
            check("preprocess_case writes files",
                  os.path.isfile(out_img) and os.path.isfile(out_msk))
            check("preprocess_case output smaller (cropped)",
                  rep2["output_shape"] < rep2["input_shape"],
                  f"{rep2['input_shape']} -> {rep2['output_shape']}")

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} tests passed.")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print("All synthetic tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
