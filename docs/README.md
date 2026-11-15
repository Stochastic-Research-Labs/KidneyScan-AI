# KidneyScan AI

Kidney tumor + cyst segmentation in 3D CT.
Like NeuroScan, but for the kidney.

## What it is

Upload a CT scan.
The model finds the kidney, tumor, and cyst in 3D.
It shows the result in a 3D viewer.
It also gives tumor and cyst volumes in cm³.

This is real AI inference.
The model runs on your input and finds the fault itself.
A normal case shows no highlight.

## Data

- Data: KiTS23
- Total cases: 599
- Training (public): 489 cases
- Test (masks private): 110 cases
- Labels: 0 = background, 1 = kidney, 2 = tumor, 3 = cyst
- Official site: https://kits-challenge.org/kits23/
- Data repo: https://github.com/neheller/kits23
- License: CC BY-NC-SA 4.0
- Free for research. Not for commercial use.

## How it works

1. CT scan is uploaded (.nii.gz).
2. CT values are clipped and normalized.
3. The scan is cropped to the kidney area.
4. A light 3D U-Net (PyTorch) runs on the scan.
5. The model returns 3 masks: kidney, tumor, cyst.
6. The viewer shows the result in 3D.
7. Volumes are shown in cm³.

## Model

- Light 3D U-Net, PyTorch.
- Small test first: 20 cases, 16 train, 4 test.
- Full train: 489 cases, 400 train, 89 validation.
- The 110 test cases are for the final check only.
- Training runs on Kaggle free GPU.

## Target Dice scores

These are targets, not results yet.

- Kidney: above 0.95
- Tumor: above 0.80
- Cyst: above 0.65

Cyst is small and hard.
Its score will be lower than the tumor score.
For reference, the KiTS19 winning Dice was:
kidney 0.974, tumor 0.851.

## How to run

### Setup

```bash
git clone https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git
cd KidneyScan-AI
pip install -r requirements.txt
```

### Train (small test)

```bash
python train.py --cases 20 --epochs 30
```

### Predict on one CT

```bash
python predict.py --input case_00001.nii.gz --output result/
```

### Web demo

- Frontend: Next.js on Vercel.
- Backend: FastAPI on Modal.
- Upload one CT file (.nii.gz).
- The demo is not live yet.

## Project status

In progress.
Phase 0 (setup) is starting now.
No final model yet. No live demo yet.

## Warning

This is an assistive tool only.
It is for learning and research.
A doctor must confirm every result.
It is NOT a final diagnosis.
Do not use it to make medical decisions.

## Who made it

Wajid Ali — NUML, BS Computer Science '28.
GitHub: https://github.com/wajidcodes
