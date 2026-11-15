# KidneyScan AI — Web demo

Kidney tumor + cyst in 3D CT (KiTS23). Like NeuroScan, but for the kidney.
Free for doctors and students.

## What it does

1. Upload one CT file (.nii.gz).
2. The model finds kidney, tumor, and cyst.
3. The 3D viewer shows two panels: original kidney (left) and prediction
   (right) — kidney grey, tumor yellow/red, cyst blue.
4. Drag to rotate 360°, scroll to zoom, right-drag to move. Reset and
   fullscreen buttons are on each panel.
5. Results page shows volumes in cm³, and Dice / Surface Dice / HD95
   when you also upload a ground-truth mask.

## Warning

Assistive tool only. A doctor must confirm the result.
This is not a diagnosis.

## Run it locally

You need Node 20+ and Python 3.12+.

### Backend (FastAPI, port 8000)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 8000
```

Check: http://127.0.0.1:8000/api/health — should say `"status": "ok"`.

### Frontend (Next.js, port 3000)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

The frontend talks to the backend through `/backend/*`, so the browser
only sees same-origin requests. No extra setup needed.

## Dummy mode vs real mode

- **Dummy (default).** The backend has no trained model yet, so it makes a
  clearly labelled TEST mask to check the UI end to end. The UI shows a
  yellow **DUMMY / TEST** badge on the viewer and results pages.
  Try it with no file: Upload page → "Dummy: tumor + cyst", or
  "Dummy: normal (no highlight)" to check that a normal case shows nothing.
- **Real mode.** Train the light 3D U-Net first (see the plan in
  `~/workspace/your_files/kits23-kidney-plan.md`), put the checkpoint at
  `backend/models_store/model.pt`, install torch + nibabel, then run with:

```bash
KIDNEYSCAN_MODE=real .venv/bin/uvicorn main:app --port 8000
```

If the checkpoint is missing, `/api/predict` returns error 503.
It never falls back to dummy silently.
To switch the frontend, pick "Real model" on the Upload page.

## API

See `docs/API_CONTRACT.md`. Main endpoints:

- `GET /api/health`
- `GET /api/demo/sample?kind=tumor|both|normal` (dummy only)
- `POST /api/predict` — multipart `file` (.nii.gz), optional `ground_truth`
  (.nii.gz), optional `mode` form field (`dummy` or `real`). `POST /predict` is an alias.

## Files

```
web/
  README.md            this file
  REUSE.md             reuse table: source, license, what changed, how to cite
  docs/API_CONTRACT.md API contract the model worker must match
  frontend/            Next.js app: home, upload, viewer, results
    app/               pages
    components/viewer/ KidneyScene (Three.js) + ComparisonViewer (2 panels)
    lib/api.ts         API client + result types
  backend/             FastAPI: /health, /predict, /demo/sample
    main.py            dummy model + real-model stub
    requirements.txt
    models_store/      put model.pt here for real mode (empty now)
```

## Reuse

See `REUSE.md`. Short version: the structure, theme, and viewer pattern
come from NeuroScan AI (same team); nnU-Net / MONAI ideas are used as an
API only, no training code copied; KiTS23 data is CC BY-NC-SA 4.0.

## Not done yet

- No trained model — real mode returns 501/503 until the model worker
  wires `real_predict()` in `backend/main.py`.
- No GitHub push, no Vercel deploy, no Modal deploy.
- No full-resolution mask download yet (the API returns a 32³ downsampled
  mask in base64 plus a per-lesion list).
