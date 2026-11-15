# KidneyScan AI

Kidney tumor + cyst segmentation in 3D CT, like NeuroScan but for kidney.

Upload a CT scan. The model finds the kidney, tumor, and cyst in 3D.
Real AI inference, no hard-coded highlight. Normal case = no highlight.

## Data

KiTS23, 599 cases total.
489 training (public), 110 test (masks private).
Labels: 0 background, 1 kidney, 2 tumor, 3 cyst.

- Official site: https://kits-challenge.org/kits23/
- Data repo: https://github.com/neheller/kits23
- License for data: CC BY-NC-SA 4.0, free for research, not commercial.
- Data is NOT in this repo. Download it with `data-prep/download.py`.

## The 6 gaps we fix

1. Cyst is very bad — Dice only 0.21 in one study, 0.18 to 0.29 with a normal U-Net.
2. Small tumors fail first — tumor Dice 0.851, human is 0.923.
3. No light model — KiTS23 winner needed 15 models on 8 big GPUs.
4. Sharp edges are weak — surface Dice 0.723 vs Dice 0.835.
5. Models drop on new hospital data — 0.85 to 0.66.
6. No free 3D demo — no paper gives doctors a simple demo.

Our model: one light 3D U-Net, 1.68M params, good at small tumors and cysts,
with sharp edges, and good on bad input. Tested one change at a time,
against an nnU-Net baseline. See `docs/literature-review.md` and
`model/ablation_plan.md`.

## Folders

- `data-prep/` — download, check, preprocess, splits (400 train / 89 val).
- `model/` — light 3D U-Net, train, inference, evaluate, Kaggle notebook.
- `web/` — Next.js frontend + FastAPI backend, 2-panel 3D viewer.
- `docs/` — literature review, deploy plan, Kaggle Phase 1 runsheet.

## How to run

Data prep:
```bash
python data-prep/download.py
python data-prep/check_data.py
python data-prep/preprocess.py
```

Train (Kaggle, 20 cases first):
```bash
python model/train_kaggle.py
```

Web demo:
```bash
cd web/backend && uvicorn main:app --reload
cd web/frontend && npm install && npm run dev
```

API: `POST /api/predict`, field `file`, returns kidney / tumor / cyst masks
and volumes in cm3. See `API_CONTRACT.md`.

## Reuse

We reuse public code efficiently, we do not rewrite it.
See `model/REUSE.md` and `web/REUSE.md` for source, license, and what changed:
nnU-Net, MONAI Auto3DSeg, Task048 weights (test only), NeuroScan pattern.

## License

Code: MIT. See `LICENSE`.
Data: CC BY-NC-SA 4.0. KiTS23 data is not in this repo.

## Warning

Assistive tool only. A doctor must confirm. Not a final diagnosis.
Learning and research only.
