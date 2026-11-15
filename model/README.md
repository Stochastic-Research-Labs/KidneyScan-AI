# KidneyScan AI — Model

Light 3D U-Net for KiTS23 kidney + tumor + cyst segmentation in 3D CT.
Labels: 0 background, 1 kidney, 2 tumor, 3 cyst.

## Files

| File | What it is |
|---|---|
| `light_unet3d.py` | Light 3D U-Net, depthwise-separable blocks. Default base 48 = 1,676,575 params (1.68M). |
| `losses.py` | Dice + cross-entropy + boundary/edge loss, cyst-aware class weights. |
| `train.py` | Training loop, AMP, checkpoint/resume, fixed 400/89 split, `--phase phase1` (20 cases, 30 epochs) / `--phase phase2` (full). |
| `inference.py` | One CT -> masks + volumes (cm3) + Dice/Surface Dice/HD95 if ground truth given. Output matches `API_CONTRACT.md`. |
| `evaluate.py` | Per-class Dice, Dice by tumor size (<4cm vs >=4cm), cyst present/absent, Surface Dice, HD95. |
| `ablation_plan.md` | Runs A-F: baseline, +light, +edge, +cyst-context, +all, nnU-Net baseline. |
| `train_kaggle.py` | Kaggle GPU runner (clone code, download KiTS23 Lite, train). |
| `kaggle_notebook.ipynb` | Same flow as a ready-to-run Kaggle notebook. |
| `API_CONTRACT.md` | Exact JSON for `POST /predict` — the web worker builds against this. |
| `REUSE.md` | Reuse table: nnU-Net, MONAI, KiTS23, 3D U-Net, BackSplit, boundary loss, NeuroScan. |

## PyTorch status (diagnosed 15 Nov 2026)

- System `python3` (3.12.3, Ubuntu) has NO PyTorch and NO nibabel.
  Only numpy 1.26.4 was present. The earlier pause note ("pip install
  blocked") did NOT reproduce: a venv install worked fine.
- Verified working install (this VM, 15 Nov 2026):
  ```
  python3 -m venv .venv
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install nibabel scipy
  ```
  Result: torch 2.15.0+cpu, nibabel 5.5.0, scipy 1.18.1.
  `py_compile` passes on all .py files; `light_unet3d.py` and
  `losses.py` smoke tests pass; end-to-end fake-data test
  (dataset -> train step -> checkpoint -> inference -> evaluate) passes.
- The venv was removed after testing (about 1 GB, must not be committed).
  Recreate it with the commands above.
- On Kaggle, PyTorch is preinstalled — use `train_kaggle.py` or
  `kaggle_notebook.ipynb`. No full training run was done here.

## Quick start (Kaggle)

```
!python train_kaggle.py --phase phase1   # 20 cases, 30 epochs, ~2-3 GPU h
!python train_kaggle.py --phase phase2 --epochs 80
```

## Quick start (this VM, after the venv install above)

```
.venv/bin/python light_unet3d.py   # param counts + forward-pass check
.venv/bin/python losses.py         # loss check on fake data
```

## Rules

- Real AI only. No hard-coded highlight or results. No invented Dice.
- Assistive only: a doctor must confirm. Not a diagnosis.
- Do not train on the 110 hidden KiTS23 test cases.
- Robustness tests live in `model/robustness/` (other worker — do not touch).
