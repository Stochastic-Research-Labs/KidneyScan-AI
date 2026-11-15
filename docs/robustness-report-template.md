# KidneyScan AI — Robustness Report Template

> Fill this in AFTER a real test run. Do not write numbers before the run.
> Every number must come from `test_robustness.py` / `external_split.py`.
> No invented Dice numbers. Empty cells mean "not tested yet", not zero.

- **Model:** _[name + checkpoint file, e.g. light 3D U-Net, `model.pt`]_
- **Date of test:** _[date]_
- **Test set:** _[how many cases, which split — internal val / pseudo-external]_
- **Severity used:** _[mild / moderate / severe — see `robustness.py` SEVERITIES]_
- **Spacing:** _[voxel spacing in mm, e.g. 1.0 x 1.0 x 1.0]_
- **Warning:** Assistive tool only. A doctor must confirm. Not a final diagnosis.

## Baseline to beat (from the literature review)

Raman et al. 2025 — a KiTS model tested on a new hospital set:

| | Tumor Dice |
|---|---|
| KiTS test | 0.85 |
| New hospital test | 0.66 |
| **Drop** | **0.19** |

**Goal for KidneyScan:** a SMALLER tumor Dice drop than 0.19 on our
pseudo-external set — and a smaller drop under each corruption below.
Raman DOI: 10.1016/j.ejrai.2025.100035. Cyst baseline is much lower:
Scalco 2024 cyst Dice 0.21; BackSplit normal U-Net cyst 0.18
(arXiv 2511.19394). Report cyst apart — never hide it in an average.

## 1. Clean vs corrupted — Dice per class

Dice 0 to 1, 1 is perfect. Drop = clean Dice minus corrupted Dice.

| Condition | Kidney Dice | Tumor Dice | Cyst Dice | Kidney drop | Tumor drop | Cyst drop |
|---|---|---|---|---|---|---|
| Clean | | | | — | — | — |
| Noise (moderate) | | | | | | |
| Low resolution (moderate) | | | | | | |
| Missing slices, 10% (moderate) | | | | | | |
| Intensity shift (moderate) | | | | | | |

Worst corruption for tumor: _[name + drop]_
Worst corruption for cyst: _[name + drop]_

## 2. Small tumors vs large tumors — tumor Dice

Small = tumor < 4 cm (equivalent sphere diameter, see `test_robustness.py`).
Large = tumor >= 4 cm. Gap 2: small tumors fail first.

| Condition | Tumor Dice, < 4 cm (n = __) | Tumor Dice, >= 4 cm (n = __) | Drop, small | Drop, large |
|---|---|---|---|---|
| Clean | | | — | — |
| Noise (moderate) | | | | |
| Low resolution (moderate) | | | | |
| Missing slices, 10% (moderate) | | | | |
| Intensity shift (moderate) | | | | |

## 3. Sharp edges — Surface Dice and HD95

Surface Dice 0 to 1, higher is better. HD95 in mm, lower is better.
Gap 4: Dice alone hides bad edges. If a cell has no lesion in both
prediction and ground truth, write "no lesion". If one side has a
lesion and the other does not, HD95 is "—" (not 0).

### Surface Dice

| Condition | Kidney | Tumor | Cyst |
|---|---|---|---|
| Clean | | | |
| Noise (moderate) | | | |
| Low resolution (moderate) | | | |
| Missing slices, 10% (moderate) | | | |
| Intensity shift (moderate) | | | |

### HD95 (mm)

| Condition | Kidney | Tumor | Cyst |
|---|---|---|---|
| Clean | | | |
| Noise (moderate) | | | |
| Low resolution (moderate) | | | |
| Missing slices, 10% (moderate) | | | |
| Intensity shift (moderate) | | | |

## 4. Pseudo-external test — simulated new hospital

**Honest limit:** this is a SIMULATED external test, not a real new
hospital. All cases are KiTS23, from the same hospitals. A small drop
here does NOT prove the model works in a new hospital.

| Set | n cases | Kidney Dice | Tumor Dice | Cyst Dice |
|---|---|---|---|---|
| Internal validation | | | | |
| Pseudo-external (never trained on) | | | | |
| **Drop (internal − external)** | — | | | |

Split seed: _[seed used in `external_split.py`]_
Did we beat Raman's tumor drop of 0.19? _[yes / no — tumor drop = __]_
Pseudo-external + mild intensity shift (extra stress): tumor Dice _[ ]_, drop _[ ]_.

## 5. Conclusion — plain English

Write 5–8 short lines. Say what held up, what broke, and what to fix next.

1. Clean scores: _[e.g. tumor Dice __, cyst Dice __]_
2. Biggest drop: _[which corruption, which class, drop __]_
3. Small tumors: _[held up / fell — drop __ vs large tumors __]_
4. Edges: _[Surface Dice / HD95 — good or bad, in one line]_
5. New-hospital simulation: _[drop __ vs Raman 0.19 — better or worse]_
6. Worst weakness: _[cyst / small tumor / edges / noise — pick one]_
7. Next fix: _[one concrete change, e.g. more cyst cases, edge loss, test-time augmentation]_
8. What this does NOT prove: _[no real hospital data tested yet]_

## 6. What was NOT tested

- [ ] Real CT from a different hospital (no such data yet)
- [ ] Severe severity (only moderate was run — say which)
- [ ] Other scanners / contrast phases
- [ ] Cases with no tumor at all (false alarms)
- [ ] Speed / GPU memory on a real demo machine

## How to fill this in

```bash
cd ~/workspace/kidneyscan-ai/model/robustness
python3 test_robustness.py        # smoke test only, NOT results
python3 external_split.py         # demo split only, NOT real cases
# Real run: call run_robustness(your_model_predict_fn, volume, truth, ...)
# in your own script, then copy the numbers into the tables above.
```
