"""KidneyScan AI — external / pseudo-new-hospital split.

The problem (Gap 5): a model can look good on its own hospital's
data and fall on a new hospital's data. Raman 2025 proved it:
a KiTS model had tumor Dice 0.85 on KiTS, but only 0.66 on a new
hospital set (drop 0.19). A hospital model did the same in reverse
(0.74 on its own data, 0.60 on KiTS).

The design here:
  * Hold out a subset of the 489 KiTS23 training cases as a
    PSEUDO-new-hospital set. Never train on it. Never tune on it.
    Test on it once, at the end, and report the Dice drop vs the
    internal validation set.
  * Suggested split of the 489 training cases (plan Phase 2 uses
    400 train / 89 validation):
        400  train
         89  internal validation (same hospital mix as training)
         --  the 89 above are NOT the external set.
    Instead, hold out the external set FIRST, from the 489:
         70  pseudo-external (about 14%, never trained on)
        419  left for train + internal validation (e.g. 340 / 79)
    Exact numbers are a choice for the training run; the code
    below takes any list of case IDs and any fraction, and checks
    that train / internal / external never overlap.
  * Optional extra stress: run the external set through a mild
    corruption too (intensity shift, see robustness.py). A real
    new hospital differs in scanner, dose and protocol — a plain
    hold-out from the same KiTS23 pool does NOT copy that.

HONEST LIMIT — read this before you quote any number:
  This is a SIMULATED external test, not a real new hospital.
  All 489 cases come from the same KiTS23 collection, the same
  hospitals and the same annotation team. A real external test
  needs CT from a different hospital, which we do not have.
  So a small drop here does NOT prove the model will work in a
  new hospital, and a big drop here is still useful: it shows
  the model is fragile even inside one dataset. Report both
  numbers (internal Dice and external Dice) and the drop, and
  always add the words "simulated external test".

Reuse: the split idea (train on one site, test on the other) is
from Raman et al. 2025, DOI 10.1016/j.ejrai.2025.100035 — no code
was published with that paper, so this code is our own. The
stratified hold-out idea follows nnU-Net v2 cross-validation
practice (Isensee et al., Nature Methods 2021, Apache 2.0):
https://github.com/MIC-DKFZ/nnUNet — we reuse the idea, not code.
"""

from __future__ import annotations

import json
import random

# Raman 2025 reference numbers (from the literature review, not ours).
# Baseline drop to beat: KiTS tumor Dice 0.85 -> hospital 0.66.
RAMAN_KITS_TUMOR_DICE = 0.85
RAMAN_HOSPITAL_TUMOR_DICE = 0.66
RAMAN_DROP = RAMAN_KITS_TUMOR_DICE - RAMAN_HOSPITAL_TUMOR_DICE  # 0.19

# Default: hold out ~14% (about 70 of 489) as pseudo-external.
DEFAULT_EXTERNAL_FRACTION = 0.14


def make_external_split(case_ids, external_fraction=DEFAULT_EXTERNAL_FRACTION,
                        seed=42, stratify_by=None):
    """Split case IDs into train / internal-validation / pseudo-external.

    case_ids: list of case IDs, e.g. ["case_00000", ...] or KiTS23
      folder names. stratify_by (optional): dict case_id -> group
      label (e.g. "has_cyst" / "no_cyst", or tumor-size group), so
      the external set keeps the same mix as the whole set.
    Returns {"train": [...], "internal_val": [...], "external": [...]}.
    The external set is held out FIRST, then the rest is split
    80/20 into train / internal validation.
    No case appears in two sets. Shuffle uses `seed`, so the split
    is repeatable — record the seed in the report.
    """
    ids = list(dict.fromkeys(case_ids))  # keep order, drop duplicates
    if len(ids) < 10:
        raise ValueError("Need at least 10 cases to make a split.")
    if not 0.0 < external_fraction < 0.5:
        raise ValueError("external_fraction must be between 0 and 0.5.")
    rng = random.Random(seed)

    if stratify_by is None:
        shuffled = ids[:]
        rng.shuffle(shuffled)
        groups = {"all": shuffled}
    else:
        groups = {}
        for cid in ids:
            groups.setdefault(str(stratify_by.get(cid, "unknown")), []).append(cid)
        for label in groups:
            rng.shuffle(groups[label])

    external, rest = [], []
    for label, members in groups.items():
        n_ext = max(1, int(round(external_fraction * len(members))))
        external.extend(members[:n_ext])
        rest.extend(members[n_ext:])
    rng.shuffle(rest)
    n_val = max(1, int(round(0.20 * len(rest))))
    internal_val = rest[:n_val]
    train = rest[n_val:]
    split = {"train": sorted(train), "internal_val": sorted(internal_val),
             "external": sorted(external)}
    check_split(split)
    return split


def check_split(split):
    """Fail if any case is in two sets, or any set is empty."""
    sets = {k: set(v) for k, v in split.items()}
    for name, s in sets.items():
        if not s:
            raise ValueError(f"Split set {name!r} is empty.")
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise ValueError(f"Cases in both {names[i]} and {names[j]}: {sorted(overlap)[:5]}")
    return True


def dice_drop_report(internal_dice, external_dice):
    """Drop per class: internal Dice minus external Dice.

    internal_dice / external_dice: dicts like {"kidney": 0.95,
    "tumor": 0.80, "cyst": 0.50}. Both must have the same classes.
    Returns {"kidney": drop, "tumor": drop, "cyst": drop, ...}.
    Positive drop = worse on the external set. Compare the tumor
    drop with RAMAN_DROP (0.19): beating Raman means a smaller drop.
    """
    if set(internal_dice) != set(external_dice):
        raise ValueError("Internal and external Dice dicts must have the same classes.")
    return {k: float(internal_dice[k]) - float(external_dice[k]) for k in internal_dice}


def save_split(split, path):
    """Save a split to JSON, with the warning that external is simulated."""
    record = {
        "warning": "SIMULATED external test only — all cases are KiTS23, "
                   "not a real new hospital. See external_split.py.",
        "n_train": len(split["train"]),
        "n_internal_val": len(split["internal_val"]),
        "n_external": len(split["external"]),
        **split,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return path


def load_split(path):
    """Load a split saved by save_split. Returns the 3 lists."""
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    split = {k: record[k] for k in ("train", "internal_val", "external")}
    check_split(split)
    return split


if __name__ == "__main__":
    # Demo with fake case IDs — shows the split works, no real data.
    fake_ids = [f"case_{i:05d}" for i in range(489)]
    fake_groups = {cid: ("has_cyst" if int(cid[-1]) % 3 == 0 else "no_cyst")
                   for cid in fake_ids}
    split = make_external_split(fake_ids, seed=42, stratify_by=fake_groups)
    print("Demo split of 489 fake cases (NOT real KiTS23 cases):")
    for k, v in split.items():
        print(f"  {k:14s} {len(v)} cases")
    print(f"Raman 2025 baseline drop to beat: tumor Dice "
          f"{RAMAN_KITS_TUMOR_DICE} -> {RAMAN_HOSPITAL_TUMOR_DICE} "
          f"(drop {RAMAN_DROP:.2f})")
    print("Example drop report (fake Dice values, NOT results):")
    print(" ", dice_drop_report(
        {"kidney": 0.96, "tumor": 0.82, "cyst": 0.50},
        {"kidney": 0.93, "tumor": 0.71, "cyst": 0.38}))
    print("OK — external_split.py runs.")
    print("REMINDER: this is a SIMULATED external test, not a real new hospital.")
