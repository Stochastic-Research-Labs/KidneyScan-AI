# KidneyScan AI — Kaggle Phase 1 Run Sheet

Date: 2026-11-15.
Status: tested on SYNTHETIC data only. No real data downloaded.
No real training done yet. No real Dice numbers exist yet.

Repo (org, NOT personal):
`https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git`
Same org as Stochastic-Research-Labs/NeuroScan-AI.
Do NOT use `wajidcodes/KidneyScan-AI`.

## Goal

Phase 1 small test. Prove the data, model and training work
before the full run.

- 20 cases: 16 train / 4 val.
- Split seed: 42 (fixed, saved in `split.json`).
- 30 epochs.
- Expected time: 2-3 GPU hours on a free Kaggle GPU (T4/P100).
- Kaggle budget: 30 GPU hours per week. Phase 1 uses ~2-3 hours.

## Kaggle setup

1. New Kaggle notebook. GPU accelerator ON. Internet ON.
2. Clone the code (notebook cell 2 does this):
   `git clone --depth 1 https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git`
3. Install: `pip install --quiet nibabel scipy huggingface_hub`
   (PyTorch is already on Kaggle.)

## Download the data — Lite route (use this one)

Notebook cell 3. Hugging Face repo: `YongchengYAO/KiTS23-Lite`.
Size: ~40.9 GB, 489 cases. License: CC BY-NC-SA 4.0.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id='YongchengYAO/KiTS23-Lite',
                  repo_type='dataset', local_dir='../data/kits23')
```

Check it worked:

```
!ls ../data/kits23/Images | wc -l   # expect 489
!ls ../data/kits23/Masks | wc -l    # expect 489
```

Lite layout is `Images/<case>.nii.gz` + `Masks/<case>.nii.gz`.
`train.py` reads this layout directly. No conversion step needed.

Do NOT use `neheller/KiTS-Challenge-Imaging` for the Lite run.
That is the FULL official set, 94.8 GB, not Lite.
(The notebook and `train_kaggle.py` used to point at it by mistake.
Fixed 2026-11-15.)

### Other route — official full download (only if Lite fails)

```bash
git clone https://github.com/neheller/kits23
cd kits23 && pip install -e .
kits23_download_data   # full 94.8 GB, all cases at once, no per-case option
```

Or use `data-prep/download.py --lite --n-cases 0 --out ./raw --yes`
for all 489 Lite cases (it has a guard: over 20 cases needs `--yes`).
Official layout is `case_XXXXX/imaging.nii.gz` +
`case_XXXXX/segmentation.nii.gz`. `train.py` reads this too.

Before training, run `data-prep/check_data.py` on the download.
It checks: image and mask shapes match, labels are only 0/1/2/3,
voxel counts per label. If it reports errors, fix the data first.

Labels: 0 background, 1 kidney, 2 tumor, 3 cyst.

## Smoke test first (notebook cell 4)

```
!python light_unet3d.py   # expect base 48 = 1,676,575 params (1.68M)
!python losses.py
```

If this fails, stop. Do not start training.

## Run Phase 1 (notebook cell 5)

```
!python train.py --data-dir ../data/kits23 --phase phase1 --out-dir checkpoints/phase1
```

Or with the runner:

```
!python train_kaggle.py --phase phase1
```

Defaults used: patch 128x128x64, batch 2, lr 1e-3, AMP on (GPU),
base channels 48, save every 10 epochs.

## Checkpoints and resume

- Saved in `checkpoints/phase1/`:
  `checkpoint_epoch10.pt`, `checkpoint_epoch20.pt`,
  `checkpoint_epoch30.pt`, `best.pt` (best val foreground Dice),
  `history.json` (one row per epoch), `split.json` (seed 42 + case lists).
- A checkpoint file is ~20 MB (model + optimizer). `best.pt` is ~7 MB.
- Download `checkpoints/` from the Kaggle output tab
  BEFORE the session ends. Kaggle sessions do not keep files.
- If the session ends early, resume in a new session.
  Same folder keeps the full history:

```
!python train.py --data-dir ../data/kits23 --phase phase1 --epochs 30 \
    --resume checkpoints/phase1/checkpoint_epoch10.pt \
    --out-dir checkpoints/phase1
```

- For Phase 2 later: same command with `--phase phase2 --epochs 80`
  and `--out-dir checkpoints/phase2`, resume from
  `checkpoints/phase2/checkpoint_epoch10.pt` (or the latest one).
  Phase 2 is 400 train / 89 val, ~12-20 GPU hours, split over sessions.

## What to record (after the run)

From `history.json` (during training) and `evaluate.py` (after):

| Metric | Where | Target (NOT a result) |
|---|---|---|
| Dice kidney | `history.json` val, then `evaluate.py` | > 0.95 |
| Dice tumor | same | > 0.80 |
| Dice cyst | same | > 0.65 |
| Surface Dice (1 mm), kidney / tumor / cyst | `evaluate.py` | higher is better |
| HD95 (mm), kidney / tumor / cyst | `evaluate.py` | lower is better |
| Dice tumor, tumor < 4 cm vs >= 4 cm | `evaluate.py` by_tumor_size | report both |
| Dice tumor + cyst, cyst present vs absent | `evaluate.py` by_cyst_present | report both |
| Params | `light_unet3d.py` | 1.68M at base 48 |
| Time per epoch, total GPU hours | `history.json` minutes | ~2-3 h for Phase 1 |

Targets are targets. Do not quote a target as a result.
A run that has not finished has no Dice.
Report mean AND n_measured per class. A class absent in both
prediction and truth is None, not 1.0 — see `evaluate.py`.

## Model size — does it fit 1 Kaggle GPU?

Yes, by design. Checked 2026-11-15:

- Base 48: 1,676,575 params = 1.68M. Weights ~6.7 MB (fp32).
  AdamW optimizer states ~13.4 MB. Base 32 = 0.77M, base 64 = 2.94M.
- Forward pass at the full patch 128x128x64, batch 1, works.
- Train.py docstring: light model + patch 128x128x64 + batch 2
  fits a Kaggle free GPU (T4/P100, 16 GB).
- If it still runs out of memory on Kaggle, do this first:
  `--patch-size 96 96 64`, then `--batch-size 1`.
  Check with `!nvidia-smi` in the notebook.

## If Dice is near zero — STOP, do not run Phase 2

Near zero after 30 epochs means the data or labels are wrong,
not that the model needs more epochs. Check in this order:

1. **Labels.** Run `check_data.py`. Mask values must be only 0/1/2/3.
   Image and mask shapes must match for every case.
2. **Look at one case.** Open one CT + mask. Is the kidney where
   the mask says it is? Are tumor/cyst blobs real or empty?
   A mask of all zeros gives Dice 0 for every class.
3. **Class weights.** Train prints them at the start, e.g.
   `[0.1, 1.0, 5.0, 5.0]` (bg, kidney, tumor, cyst).
   If tumor/cyst weights are 0 or missing, the model ignores them.
4. **Loss falling?** In `history.json`, `train_loss` must go down
   over epochs. If it stays flat, the learning rate or the data
   loading is broken — check the batch (image + mask) visually.
5. **Train vs val.** If train Dice rises but val Dice stays ~0,
   the split may be wrong or the val cases may be bad data.
   Check `split.json` (seed 42, 16 train / 4 val, no overlap).
6. **Crop / HU window.** Train clips CT to [-200, 300] HU and crops
   to the mask bounding box. If a case has no mask voxels, the crop
   is skipped and the whole volume is used — check that case.
7. **One-epoch overfit test.** Train 1 case for a few epochs.
   If it cannot overfit 1 case, the model/loss code is broken.
   If it can, the problem is the data, not the code.

Only start Phase 2 when Phase 1 val Dice is clearly above zero
and rising (kidney should move first, then tumor, cyst last).

## Test evidence (local, 2026-11-15, synthetic data only)

Test venv: torch 2.15.0+cpu, nibabel 5.5.0, scipy 1.18.1.
No real data was downloaded. No real training was done.

- `py_compile` passes on all `model/*.py` and `data-prep/*.py`.
- All notebook code cells `py_compile` OK.
- `light_unet3d.py` smoke test passes: base 32 = 0.77M,
  base 48 = 1.68M, base 64 = 2.94M, forward pass works.
- `losses.py` smoke test passes.
- Built 20 synthetic cases (32x32x16 fake CT + kidney/tumor/cyst
  blobs) in BOTH layouts: official `case_XXXXX/` and Lite
  `Images/` + `Masks/`.
- `train.py --phase phase1 --epochs 2` (small patch, CPU) passes
  on BOTH layouts: 16 train / 4 val, loss falls, checkpoints +
  `best.pt` + `history.json` + `split.json` all written.
- `--resume` passes, including resume into the same out-dir
  (history keeps epochs 1, 2, 3).
- `inference.py --fast` passes on one synthetic case.
- `evaluate.py` passes on one synthetic prediction, report saved.
- Synthetic numbers (loss ~2.05 to ~1.62, val Dice ~0.08 to ~0.13
  after only 2 epochs on fake data) are NOT results. Ignore them.

## Errors found and fixed in this check

1. **Wrong dataset repo (serious).** Notebook cell 3 and
   `train_kaggle.py` said "Lite 40.9 GB" but downloaded
   `neheller/KiTS-Challenge-Imaging` — the FULL 94.8 GB set.
   Fixed to `YongchengYAO/KiTS23-Lite` in both files.
2. **Train only read the official layout.** `train.py` `list_cases()`
   only found `case_XXXXX/imaging.nii.gz`. A Lite download
   (`Images/` + `Masks/`) would have found 0 cases.
   Fixed: `list_cases()` now reads official, Lite and nnU-Net
   layouts. `train_kaggle.py` presence check fixed the same way.
3. **Training crashed at the end of epoch 1.** `train_one_epoch()`
   and `validate()` did `{k: v ... for k, v in totals}` — iterating
   a dict gives keys only, so it crashed with
   `ValueError: too many values to unpack`. Fixed to
   `totals.items()` in both functions. This means `train.py` had
   never run end-to-end before this check.
4. **Phase 1 never saved `split.json`.** `make_split()` returned
   early for phase1 before writing the file. Fixed: the split file
   is now written for both phases.
5. **Resume wiped history.** A resumed run overwrote `history.json`
   with only the new epochs. Fixed: resuming into the same out-dir
   keeps earlier epochs.
6. **Notebook re-run would fail.** `git clone` with no guard fails
   if the folder already exists. Fixed with an
   `if [ ! -d KidneyScan-AI ]` guard. Also added `huggingface_hub`
   to the pip install (cell 3 imports it) and fixed the cell 7
   example paths for the Lite layout.
7. **Repo URL.** Notebook and `train_kaggle.py` cloned
   `wajidcodes/KidneyScan-AI`. Fixed to the org repo
   `https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git`.

## NOT tested yet (blocked / out of scope)

- Full notebook execute (`jupyter nbconvert --execute`) was NOT run
  locally: cells 2-3 need a GitHub clone and a 40.9 GB download.
  Code cells were compiled and the training code was run on
  synthetic data instead (see above).
- No real KiTS23 download (40.9 GB Lite / 94.8 GB full) — not done here.
- No real GPU run, no 30-epoch run. First real run is on Kaggle.
- No push, deploy or post. Nothing was pushed to GitHub.
