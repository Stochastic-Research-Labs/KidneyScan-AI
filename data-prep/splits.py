#!/usr/bin/env python3
"""Train/validation split for KidneyScan AI (KiTS23).

Plan: 400 train / 89 validation out of 489 training cases.
Stratified by TWO things, so both splits look the same:
  * tumor size: <4 cm vs >=4 cm vs no tumor
  * cyst: present vs absent
Seed is fixed (42) so the split can be reproduced exactly.
The 110 KiTS23 test cases are NEVER used here — they stay for the final
check only (their masks are private anyway).

Tumor size: we do not have a measured max diameter in the data check,
so size is an equivalent-sphere diameter from tumor volume:
    d = 2 * (3V / 4*pi) ** (1/3),  V in cm3.
This is an APPROXIMATION, written in the output so nobody mistakes it
for a radiologist's measurement. It is only used to balance the split
and to group results, never as a medical claim.

Also in this file: size-stratified evaluation helpers.
  * dice_score(pred, truth) — per-class Dice, 0 to 1.
  * stratified_report(case_results) — groups per-case Dice by tumor-size
    group and cyst presence, so gaps 1 (cyst) and 2 (small tumors) can be
    seen separately instead of hidden in one average Dice.
The model worker calls these on real predictions. No Dice numbers are
hard-coded or invented here.

Usage:
    python3 splits.py --stats ./data_check.json --out ./splits
    python3 splits.py --stats ./data_check.json --out ./splits --seed 42

Input --stats is the JSON from check_data.py (per-case label counts and
voxel spacing). Output: train.json, val.json, split_summary.json.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

SEED = 42
N_TRAIN_FULL = 400
N_VAL_FULL = 89
TOTAL_FULL = N_TRAIN_FULL + N_VAL_FULL  # 489
SIZE_CUTOFF_CM = 4.0


def tumor_size_group(tumor_volume_cm3):
    """'none' / 'small' (<4 cm) / 'large' (>=4 cm), by equivalent-sphere diameter."""
    if tumor_volume_cm3 is None or tumor_volume_cm3 <= 0:
        return "none"
    diameter = 2.0 * (3.0 * tumor_volume_cm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    return "small" if diameter < SIZE_CUTOFF_CM else "large"


def dice_score(pred, truth):
    """Dice overlap, 0 to 1. Empty vs empty = 1.0 (nothing to miss)."""
    import numpy as np

    pred = np.asarray(pred).astype(bool)
    truth = np.asarray(truth).astype(bool)
    inter = int((pred & truth).sum())
    total = int(pred.sum()) + int(truth.sum())
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def per_class_dice(pred_mask, true_mask, labels=(1, 2, 3)):
    """Dice per label. Returns {label: dice}. Labels: 1 kidney, 2 tumor, 3 cyst."""
    return {int(lbl): dice_score(pred_mask == lbl, true_mask == lbl)
            for lbl in labels}


def stratified_report(case_results):
    """Group per-case Dice by tumor size and cyst presence.

    case_results: list of dicts with keys:
        case, tumor_size_group ("small"/"large"/"none"),
        cyst_present (bool), dice (dict label -> float).
    Returns {group_key: {label: mean Dice, "n": count}}.
    Mean of nothing is reported as None, never as 0 or 1.
    """
    groups = defaultdict(list)
    for r in case_results:
        key = (r["tumor_size_group"],
               "cyst" if r["cyst_present"] else "no_cyst")
        groups[key].append(r)
    out = {}
    for key in sorted(groups):
        rows = groups[key]
        entry = {"n": len(rows)}
        for lbl in (1, 2, 3):
            vals = [r["dice"][lbl] for r in rows if lbl in r["dice"]]
            entry[f"dice_label_{lbl}"] = (sum(vals) / len(vals)) if vals else None
        out[f"{key[0]}__{key[1]}"] = entry
    return out


def case_stratum(case):
    """Stratum key for one case dict from check_data.py output."""
    vols = case.get("volumes_cm3") or {}
    counts = case.get("label_counts") or {}
    tumor_vol = vols.get("tumor")
    if tumor_vol is None and counts.get("tumor"):
        spacing = case.get("voxel_spacing_mm")
        if spacing:
            tumor_vol = counts["tumor"] * spacing[0] * spacing[1] * spacing[2] / 1000.0
    size = tumor_size_group(tumor_vol)
    cyst = bool((counts.get("cyst") or 0) > 0)
    return (size, "cyst" if cyst else "no_cyst")


def make_split(cases, n_train=N_TRAIN_FULL, n_val=N_VAL_FULL, seed=SEED):
    """Stratified split. Returns (train_ids, val_ids, summary dict).

    Cases are shuffled inside each stratum with the fixed seed, then
    dealt to train/val in proportion to the stratum size. If fewer than
    489 cases are given (e.g. a 5-case test), the 400/89 numbers are
    scaled down in the same ratio and the summary says so.
    """
    total = len(cases)
    if total == 0:
        raise ValueError("no cases to split")
    if total < TOTAL_FULL:
        n_train = max(1, round(total * N_TRAIN_FULL / TOTAL_FULL))
        n_val = total - n_train
    by_stratum = defaultdict(list)
    for c in cases:
        by_stratum[case_stratum(c)].append(c["case"])
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(by_stratum):
        ids = sorted(by_stratum[key])
        rng.shuffle(ids)
        # Train share inside this stratum, same ratio as the whole split.
        k = round(len(ids) * n_train / total)
        k = max(0, min(len(ids), k))
        train.extend(ids[:k])
        val.extend(ids[k:])
    # Fix rounding so totals match exactly.
    rng.shuffle(train)
    rng.shuffle(val)
    while len(train) > n_train and train:
        val.append(train.pop())
    while len(val) > n_val and val:
        train.append(val.pop())
    while len(train) < n_train and val:
        train.append(val.pop(0))
    while len(val) < n_val and train:
        val.append(train.pop(0))
    summary = {
        "seed": seed,
        "n_total": total,
        "n_train": len(train),
        "n_val": len(val),
        "requested_train_val": [n_train, n_val],
        "scaled_down": total < TOTAL_FULL,
        "size_cutoff_cm": SIZE_CUTOFF_CM,
        "size_note": "Tumor size is an equivalent-sphere diameter from "
                     "tumor volume, an approximation for balancing only.",
        "strata": {},
    }
    for key in sorted(by_stratum):
        ids = set(by_stratum[key])
        summary["strata"][f"{key[0]}__{key[1]}"] = {
            "total": len(ids),
            "train": len([i for i in train if i in ids]),
            "val": len([i for i in val if i in ids]),
        }
    return sorted(train), sorted(val), summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stratified 400/89 split for KiTS23.")
    ap.add_argument("--stats", required=True,
                    help="data_check.json from check_data.py")
    ap.add_argument("--out", required=True, help="Output folder for the split JSONs.")
    ap.add_argument("--seed", type=int, default=SEED, help="Fixed seed. Default: 42.")
    ap.add_argument("--n-train", type=int, default=N_TRAIN_FULL)
    ap.add_argument("--n-val", type=int, default=N_VAL_FULL)
    args = ap.parse_args(argv)

    try:
        with open(args.stats, encoding="utf-8") as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: stats file not found: {args.stats}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: stats file is not valid JSON: {exc}", file=sys.stderr)
        return 2
    cases = stats.get("cases", [])
    # Keep only cases that loaded cleanly and have a mask.
    usable = [c for c in cases if not c.get("errors") and c.get("label_counts")]
    skipped = len(cases) - len(usable)
    if skipped:
        print(f"WARNING: skipped {skipped} cases with errors or no mask.", file=sys.stderr)
    if not usable:
        print("ERROR: no usable cases in stats file.", file=sys.stderr)
        return 1
    train, val, summary = make_split(usable, args.n_train, args.n_val, args.seed)
    assert not (set(train) & set(val)), "train/val overlap — this must never happen"
    os.makedirs(args.out, exist_ok=True)
    for name, ids in (("train.json", train), ("val.json", val)):
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
    with open(os.path.join(args.out, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Split done: {len(train)} train, {len(val)} val, seed {args.seed}.")
    for key, s in summary["strata"].items():
        print(f"  {key}: total {s['total']}, train {s['train']}, val {s['val']}")
    print(f"Files: {args.out}/train.json, val.json, split_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
