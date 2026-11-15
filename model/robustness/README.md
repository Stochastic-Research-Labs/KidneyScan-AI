# KidneyScan AI — Robustness Tests

Short version: we make a clean CT worse on purpose (noise, low
resolution, missing slices, different brightness), then check how
much the model score falls. A good model falls a little. A weak
model falls a lot.

**Warning:** assistive tool only. A doctor must confirm.
Not a final diagnosis.

## Files

| File | What it does |
|---|---|
| `robustness.py` | 4 corruptions for TEST ONLY: noise, low resolution, drop 10% of slices, intensity shift. Never use for training. |
| `test_robustness.py` | Runs a model on clean vs corrupted CT. Scores Dice, Surface Dice, HD95, and Dice drop, per class and per tumor size (< 4 cm vs >= 4 cm). |
| `external_split.py` | Holds out some cases as a fake "new hospital" set. Never train on it. Reports the Dice drop vs internal validation. |

## How to run

```bash
cd ~/workspace/kidneyscan-ai/model/robustness
python3 -m py_compile robustness.py test_robustness.py external_split.py
python3 test_robustness.py    # smoke test only — scores are NOT results
python3 external_split.py     # demo split only — fake case IDs, NOT real data
```

Real run (in your own script):

```python
from test_robustness import run_robustness

# predict_fn: your trained model. It takes a CT volume and
# returns a mask (0 background, 1 kidney, 2 tumor, 3 cyst).
results = run_robustness(predict_fn, volume, ground_truth,
                         spacing_mm=(1.0, 1.0, 1.0),
                         severity="moderate")
```

Then copy the numbers into `docs/robustness-report-template.md`.

You need only NumPy and SciPy. No torch, no MONAI, no GPU.

## Baseline to beat

Raman 2025: tumor Dice 0.85 on KiTS, 0.66 on a new hospital.
Drop = 0.19. Our goal: a smaller drop than 0.19.
Cyst is much harder: Scalco 2024 cyst Dice 0.21.

## Reuse table

For each reused code / idea: source, license, what we changed, how to cite.

| Reused idea | Source | License | What we changed and why | How to cite |
|---|---|---|---|---|
| Noise / low-res / intensity-shift test idea (from nnU-Net v2 augmentation + evaluation ideas) | https://github.com/MIC-DKFZ/nnUNet | Apache 2.0 | Rewrote as small NumPy/SciPy test-only functions, no torch, no nnU-Net install; severities set for KiTS23 CT; drop-slices test is our own addition | Isensee et al., Nature Methods 2021, DOI 10.1038/s41592-020-01008-z |
| Intensity / noise transform idea | https://github.com/Project-MONAI/MONAI | Apache 2.0 | Rewrote as plain functions on NumPy arrays, not MONAI transform classes, so no MONAI install is needed | MONAI Consortium, https://monai.io |
| External-test design (train on one site, test on the other) | Raman et al. 2025, DOI 10.1016/j.ejrai.2025.100035 | — (paper, no code published) | No code reused; only the test design and the 0.85 → 0.66 numbers as the baseline to beat | Raman et al., Eur J Radiol Artif Intell 2025, 3:100035 |
| Cyst is the weak class; score it apart | Scalco et al. 2024, DOI 10.1002/mp.17189; Saluja et al. BackSplit, arXiv 2511.19394 | — (papers, no code reused) | No code reused; only the evidence (cyst Dice 0.21 / 0.18) that shaped the per-class tables | Scalco et al., Med Phys 2024; Saluja et al., arXiv 2511.19394 |
| Surface Dice / HD95 as edge scores | KiTS23 winner report, arXiv 2310.04110 (surface Dice 0.723) | — (paper) | Rewrote both metrics with SciPy distance transforms; no code copied | Myronenko et al., arXiv 2310.04110 |

Full detail is in the REUSE notes inside each code file and in
`docs/literature-review.md`.

## What was NOT tested

- No real hospital data. The "external" set is a hold-out from the
  same KiTS23 data — a simulation, not a real new hospital.
- No real trained model yet. The smoke test uses a fake predictor
  and fake CT. Its scores mean nothing — do not quote them.
- No severe severity run, no other scanners, no contrast phases,
  no cases with no tumor (false alarms).
- No speed or GPU memory test.
