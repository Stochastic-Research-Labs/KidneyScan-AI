#!/usr/bin/env python3
"""Check KiTS23 data files and report what is really there.

Labels (KiTS23): 0 background, 1 kidney, 2 tumor, 3 cyst.
Expected image shape: 512 x 512 in x/y, slice count (z) varies by case.

Finds case pairs in any of these layouts (no layout is assumed):
  * Official kits23:  <data>/case_00000/imaging.nii.gz
                       <data>/case_00000/segmentation.nii.gz
  * Lite mirror:      <data>/Images/<case>.nii.gz
                       <data>/Masks/<case>.nii.gz
  * nnU-Net style:    <data>/imagesTr/<case>_0000.nii.gz
                       <data>/labelsTr/<case>.nii.gz

For each case it checks: files exist, image and mask shapes match,
x/y are 512x512, mask labels are only 0/1/2/3, voxel counts per label,
voxel spacing from the NIfTI header. Results go to JSON + a text report.

Usage:
    python3 check_data.py --data ./raw --out ./data_check.json
    python3 check_data.py --data ./raw/dataset --out ./data_check.json

Needs: pip install nibabel numpy
Exit code: 0 = all cases OK, 1 = missing or bad files found,
2 = nibabel/numpy not installed.
"""
import argparse
import json
import os
import sys
from collections import Counter

LABELS = {0: "background", 1: "kidney", 2: "tumor", 3: "cyst"}
EXPECTED_XY = (512, 512)


def _stem(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        name = name[: -len(".nii.gz")]
    if name.endswith("_0000"):
        name = name[: -len("_0000")]
    return name


def find_pairs(data_dir):
    """Return {case_id: (image_path, mask_path_or_None)} for all layouts."""
    pairs = {}

    def add(img, msk, case=None):
        pairs[case or _stem(img)] = (img, msk)

    # Official layout: case_*/imaging.nii.gz + segmentation.nii.gz
    for entry in sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else []:
        case_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(case_dir):
            continue
        img = os.path.join(case_dir, "imaging.nii.gz")
        msk = os.path.join(case_dir, "segmentation.nii.gz")
        if os.path.isfile(img):
            add(img, msk if os.path.isfile(msk) else None, case=entry)

    # Flat Images/Masks layout (Lite mirror)
    for img_dir, msk_dir in (("Images", "Masks"), ("imagesTr", "labelsTr"),
                             ("images", "masks")):
        img_d = os.path.join(data_dir, img_dir)
        msk_d = os.path.join(data_dir, msk_dir)
        if not os.path.isdir(img_d):
            continue
        for name in sorted(os.listdir(img_d)):
            if not name.endswith(".nii.gz"):
                continue
            img = os.path.join(img_d, name)
            stem = _stem(name)
            msk = None
            if os.path.isdir(msk_d):
                for cand in (stem + ".nii.gz", stem + "_0000.nii.gz"):
                    if os.path.isfile(os.path.join(msk_d, cand)):
                        msk = os.path.join(msk_d, cand)
                        break
            if _stem(img) not in pairs:
                add(img, msk)

    # Flat folder of .nii.gz files (image + mask side by side, rare)
    if not pairs and os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.endswith(".nii.gz") and "seg" not in name.lower():
                add(os.path.join(data_dir, name), None)
    return pairs


def check_case(case_id, img_path, msk_path):
    """Check one case. Returns a dict; 'errors' is empty when the case is OK."""
    import nibabel as nib
    import numpy as np

    out = {"case": case_id, "image": img_path, "mask": msk_path,
           "errors": [], "warnings": []}
    img = nib.load(img_path)
    data = np.asanyarray(img.dataobj)
    out["image_shape"] = [int(s) for s in data.shape]
    out["image_dtype"] = str(data.dtype)
    out["voxel_spacing_mm"] = [round(float(s), 3) for s in img.header.get_zooms()[:3]]
    if len(data.shape) != 3:
        out["errors"].append(f"image is not 3D, shape {data.shape}")
        return out
    if tuple(data.shape[:2]) != EXPECTED_XY:
        out["warnings"].append(
            f"image x/y is {data.shape[0]}x{data.shape[1]}, expected 512x512")
    if msk_path is None:
        out["errors"].append("mask file missing")
        return out
    msk = nib.load(msk_path)
    mdata = np.asanyarray(msk.dataobj)
    out["mask_shape"] = [int(s) for s in mdata.shape]
    if mdata.shape != data.shape:
        out["errors"].append(
            f"mask shape {mdata.shape} != image shape {data.shape}")
        return out
    values, counts = np.unique(mdata, return_counts=True)
    out["label_counts"] = {LABELS.get(int(v), f"unknown_{int(v)}"): int(c)
                           for v, c in zip(values, counts)}
    bad = [int(v) for v in values if int(v) not in LABELS]
    if bad:
        out["errors"].append(f"mask has unknown labels: {bad}")
    spacing = out["voxel_spacing_mm"]
    vox_vol_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    out["voxel_volume_cm3"] = round(vox_vol_cm3, 6)
    out["volumes_cm3"] = {
        name: round(int(out["label_counts"].get(name, 0)) * vox_vol_cm3, 2)
        for name in ("kidney", "tumor", "cyst")
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verify KiTS23 .nii.gz files.")
    ap.add_argument("--data", required=True, help="Folder with the cases.")
    ap.add_argument("--out", default=None,
                    help="JSON report path. Default: <data>/../data_check.json "
                         "or ./data_check.json next to this script.")
    args = ap.parse_args(argv)

    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("ERROR: nibabel and numpy are needed.", file=sys.stderr)
        print("Run: pip install nibabel numpy", file=sys.stderr)
        return 2

    if not os.path.isdir(args.data):
        print(f"ERROR: data folder not found: {args.data}", file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    out_json = args.out or os.path.join(here, "data_check.json")
    out_txt = out_json[:-5] + ".txt" if out_json.endswith(".json") else out_json + ".txt"

    pairs = find_pairs(args.data)
    report = {"dataset": "KiTS23", "data_dir": os.path.abspath(args.data),
              "n_cases_found": len(pairs), "cases": [],
              "missing_masks": [], "errors_total": 0}
    for case_id in sorted(pairs):
        img, msk = pairs[case_id]
        if msk is None:
            report["missing_masks"].append(case_id)
        try:
            res = check_case(case_id, img, msk)
        except Exception as exc:  # corrupt file: record it, keep going
            res = {"case": case_id, "image": img, "mask": msk,
                   "errors": [f"could not read files: {exc}"], "warnings": []}
        report["cases"].append(res)
        report["errors_total"] += len(res["errors"])

    # Totals across cases (only cases that loaded cleanly)
    totals = Counter()
    n_512 = n_cyst = n_tumor = 0
    for c in report["cases"]:
        if c.get("label_counts"):
            for k, v in c["label_counts"].items():
                totals[k] += v
            if c["label_counts"].get("cyst", 0) > 0:
                n_cyst += 1
            if c["label_counts"].get("tumor", 0) > 0:
                n_tumor += 1
        if c.get("image_shape", [])[:2] == [512, 512]:
            n_512 += 1
    report["summary"] = {
        "cases_with_512x512": n_512,
        "cases_with_tumor": n_tumor,
        "cases_with_cyst": n_cyst,
        "cases_missing_mask": len(report["missing_masks"]),
        "total_label_voxels": dict(totals),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    lines = [
        "KiTS23 data check",
        f"Data folder: {report['data_dir']}",
        f"Cases found: {report['n_cases_found']}",
        f"Cases 512x512: {n_512}",
        f"Cases with tumor: {n_tumor}, with cyst: {n_cyst}",
        f"Cases missing mask: {len(report['missing_masks'])}",
        f"Total errors: {report['errors_total']}",
        f"Total label voxels: {dict(totals)}",
    ]
    if report["missing_masks"]:
        lines.append("Missing masks: " + ", ".join(report["missing_masks"]))
    for c in report["cases"]:
        for e in c["errors"]:
            lines.append(f"ERROR {c['case']}: {e}")
        for w in c["warnings"]:
            lines.append(f"WARN {c['case']}: {w}")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"JSON report: {out_json}")
    print(f"Text report: {out_txt}")
    return 1 if report["errors_total"] or report["missing_masks"] else 0


if __name__ == "__main__":
    sys.exit(main())
