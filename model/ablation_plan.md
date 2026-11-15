# KidneyScan AI — Ablation Plan

Date: 15 Nov 2026. Data: KiTS23, 489 public training cases.
Split (FIXED for every run): 400 train / 89 validation, seed 42.
The 110 hidden KiTS23 test cases are never used here.
Metrics (every run reports all of these, per class):
Dice (kidney / tumor / cyst), Surface Dice (1 mm), HD95 (mm),
Dice by tumor size (<4 cm vs >=4 cm), Dice by cyst present/absent,
params (millions), inference time per case, GPU memory.

## Runs (each is one train.py run with flags)

| # | Run | What changes vs baseline | What it proves |
|---|---|---|---|
| A | Baseline | Light 3D U-Net (base 16), Dice+CE loss, no edge loss, cyst trained as a normal 4th class, no special cyst weighting | The floor. Everything else must beat this on the same 89 cases. |
| B | +light only | Same as A but base_channels 32 vs 48 vs 64 (param count + speed test). Loss unchanged. | Whether a smaller model keeps the Dice. Gap 3: light model. If base 32 loses <0.02 Dice vs base 48, ship base 32. |
| C | +edge only | A + boundary loss (w_edge 0.5). No other change. | Whether edge loss improves Surface Dice + HD95 without hurting Dice. Gap 4: sharp edges. |
| D | +cyst-context only | A + cyst class weight raised (5.0) and cyst oversampling: cases with cyst sampled 2x as often. No edge loss. | Whether treating cyst as its own weighted class (BackSplit idea, simple form) lifts cyst Dice from the 0.18-0.29 baseline. Gap 1. |
| E | +all | B-best base channels + edge loss + cyst weighting/oversampling. | The full KidneyScan model. Gap 1+3+4 together. Must beat A on cyst Dice AND Surface Dice, at similar or lower params. |
| F | nnU-Net baseline | nnU-Net v2 3D full-res, same 400/89 split, default config. No KidneyScan changes. | Fair external comparison. Do NOT compare our 89-case Dice with the KiTS23 winners' hidden-test Dice (different cases). Gap: is light + edge close to heavy? |

## Extra splits (run on the best model, E, and on A for comparison)

- Tumor size: Dice tumor for <4 cm vs >=4 cm (Gap 2). Expect small worse.
  Report both; a fix that only helps big tumors is not enough.
- Cyst present vs absent: Dice tumor + cyst for both groups (Gap 1).
  Cyst-present cases are where tumor/cyst confusion happens.
- Robustness (Gap 5, done by the robustness worker in model/robustness/):
  clean vs +noise vs low resolution vs 10% slices dropped. Report Dice
  drop for A vs E. A good model falls less. This worker does NOT run
  those tests; the robustness worker owns them.

## Rules (fair comparison)

1. Same split, same seed, same patch size, same epochs for A-E.
   Only the named change differs. Changing two things at once proves nothing.
2. Same metrics, same code (evaluate.py) for every run.
3. Report mean AND n_measured per class. A class absent in both
   prediction and truth is None, not 1.0 — see evaluate.py.
4. No invented numbers. A run that has not finished has no Dice.
   Targets from the plan (kidney >0.95, tumor >0.80, cyst >0.65) are
   targets, not results. Quote them as targets until measured.
5. Phase 1 first: run A on 20 cases / 30 epochs. If Dice is near zero,
   fix data/labels before any full run.
6. Save every run: checkpoint + history.json + eval_report.json, named
   by run letter (e.g. checkpoints/run_A/best.pt).

## Decision rules

- Ship E only if it beats A on cyst Dice AND Surface Dice, with params
  <=5M and no worse than 0.02 Dice drop on kidney.
- If C (edge only) hurts Dice, keep edge weight low (0.25) and re-test;
  do not drop the edge idea on one bad run.
- If D lifts cyst Dice but drops tumor Dice (confusion), try a lower
  cyst weight (3.0) before giving up.
- If F (nnU-Net) beats E by >0.05 tumor Dice, say so honestly in the
  paper: light trades a little Dice for speed/size. Do not hide it.
