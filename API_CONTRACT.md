# KidneyScan AI — API Contract (canonical, v1.1)

Version: 1.1 — 15 Nov 2026. Owner: Model worker + Web worker.
**This file is the single source of truth.** The old copies at
`model/API_CONTRACT.md` and `web/docs/API_CONTRACT.md` are pointers to
this file — do not edit them, edit this one. `model/inference.py`
(`predict()`) and `web/backend/main.py` both implement exactly this shape.

## How the two old contracts were merged

| Item | Model contract said | Web contract said | Canonical (this file) |
|---|---|---|---|
| Endpoint | `POST /predict` | `POST /api/predict` | `POST /api/predict` (primary) + `POST /predict` alias, same handler |
| File field | `file` | `ct` | `file` |
| Mask | `.nii.gz` file only, JSON never embeds voxels | `mask_base64` 32³ preview only | Both, with clear roles: `mask.file` = full-res `.nii.gz` download; `mask_base64` = 32³ viewer preview only |
| Finding | `finding` string (4 exact values) | `findings` object | Both: `finding` string + `findings` object |
| Model | `model` object | `model` string | `model` object (+ top-level `mode` / `is_dummy` kept) |
| Scores | Per class **including background** | Kidney / tumor / cyst only | Kidney / tumor / cyst only — **no background** |
| Errors | 400 / 500 | 400 / 413 / 501 / 503 | **400 / 413 / 503 only** (501 gone: real inference is now wired; 500 gone: a failed real model is 503) |

Why `/api/predict` is primary: the frontend already calls it through the
Next.js proxy `/backend/*`, and it matches the NeuroScan pattern
(`POST /api/...`). The bare `/predict` alias exists so the model
contract's literal path also works.

## Endpoints

Backend: FastAPI. Base URL local: `http://127.0.0.1:8000`.
Frontend talks to it through the Next.js proxy `/backend/*`
(see `frontend/next.config.ts`), so the browser only sees same-origin
requests — same pattern as NeuroScan.

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | — | `{name, docs, health}` |
| GET | `/api/health` | — | `{status, version, mode, model_loaded, checkpoint_present}` |
| GET | `/api/demo/sample?kind=tumor\|both\|normal` | — | `PredictionResult`, always `mode: "dummy"` |
| POST | `/api/predict` (alias: `/predict`) | multipart: `file` (.nii.gz, required), `ground_truth` (.nii.gz, optional), `mode` form field (`dummy`\|`real`, optional, defaults to `KIDNEYSCAN_MODE`) | `PredictionResult` |
| GET | `/api/predict/{case_id}/mask` (alias: `/predict/{case_id}/mask`) | — | Full-resolution predicted mask as a `.nii.gz` file download |

No other fields. No patient name, no ID, no metadata is accepted or stored.

## PredictionResult (JSON) — success, HTTP 200

```json
{
  "status": "ok",
  "case_id": "case_00000",
  "mode": "dummy",
  "is_dummy": true,
  "model": {
    "name": "dummy-test-mask",
    "checkpoint": null,
    "base_channels": 48,
    "num_classes": 4,
    "classes": ["background", "kidney", "tumor", "cyst"]
  },
  "note": "DUMMY / TEST prediction. Generated for UI testing only. Not a real model result — switch to real mode for real inference.",
  "input": {
    "file": "case_00000_imaging.nii.gz",
    "shape": [512, 512, 120],
    "spacing_mm": [0.78, 0.78, 3.0]
  },
  "shape": [512, 512, 120],
  "spacing_mm": [0.78, 0.78, 3.0],
  "finding": "tumor and cyst detected",
  "findings": {
    "tumor_present": true,
    "cyst_present": true,
    "cyst_count": 1,
    "tumor_volume_cm3": 18.20,
    "cyst_volume_cm3": 4.10
  },
  "volumes_cm3": {"kidney": 312.45, "tumor": 18.20, "cyst": 4.10},
  "voxel_counts": {"kidney": 171234, "tumor": 9978, "cyst": 2248},
  "lesions": [
    {"class": "tumor", "center_xyz": [0.5, 0.5, 0.6], "radius_mm": 6.6, "volume_cm3": 18.20}
  ],
  "mask": {
    "encoding": "nifti-gz file, same shape/orientation as input; values 0 background, 1 kidney, 2 tumor, 3 cyst",
    "file": "/api/predict/case_00000/mask",
    "shape": [512, 512, 120]
  },
  "mask_grid": 32,
  "mask_base64": "<base64 uint8 mask, 32x32x32, row-major D,H,W, labels 0/1/2/3>",
  "labels": {"0": "background", "1": "kidney", "2": "tumor", "3": "cyst"},
  "scores": null,
  "inference_seconds": 23.4,
  "warning": "Assistive tool only. This is not a diagnosis. A doctor must confirm the result."
}
```

Notes on the example values above: they are EXAMPLES of the shape only,
not measured results. No real Dice is quoted until `evaluate.py` has run
on real data.

### Which fields come from where

- **Core fields — identical on both sides.** `model/inference.py`
  `predict()` returns all of these, and the backend returns them unchanged:
  `status`, `case_id`, `mode`, `is_dummy`, `model`, `note`, `input`,
  `shape`, `spacing_mm`, `finding`, `findings`, `volumes_cm3`,
  `voxel_counts`, `mask`, `scores`, `inference_seconds`, `warning`.
- **Web-only additions** (backend adds them for the viewer; the model
  worker does not produce them): `lesions`, `mask_grid`, `mask_base64`,
  `labels`.

### `finding` values (exact strings)

| Value | Meaning |
|---|---|
| `"no tumor or cyst detected"` | Tumor voxels = 0 and cyst voxels = 0. Normal case: no highlight. |
| `"tumor detected, no cyst detected"` | Tumor > 0, cyst = 0. |
| `"cyst detected, no tumor detected"` | Cyst > 0, tumor = 0. |
| `"tumor and cyst detected"` | Both > 0. |

The finding is computed from the model output only — never hard-coded
from the file name or any input field. Upload selects the INPUT only.
The model decides the output.

### Mask

- Labels: 0 background, 1 kidney, 2 tumor, 3 cyst (KiTS23).
- The full-resolution predicted mask is a `.nii.gz` file, same shape,
  spacing and orientation (affine) as the input CT.
- `mask.file` is the download URL (`/api/predict/{case_id}/mask`).
  `mask.file` is `null` only when the mask file could not be written
  (dummy mode without nibabel installed — dummy must work with no
  torch/nibabel). The viewer then uses the `mask_base64` preview.
- `mask_base64` is the mask downsampled to 32×32×32, base64 uint8,
  row-major D,H,W — a viewer preview only, never the full mask.
- The web viewer colours: kidney grey `#9aa7b0`, tumor yellow/red
  `#facc15` (viewer also uses `#ef4444` for the tumor core), cyst blue
  `#38bdf8`. Same values in `frontend/lib/segmentation-colors.ts`
  (and `frontend/lib/api.ts` `CLASS_COLORS`) and `backend/main.py`
  `CLASS_INFO`.

### Volumes

- `volumes_cm3` = voxel count × voxel volume (from NIfTI spacing),
  divided by 1000 (mm³ → cm³), rounded to 2 decimals.
- Kidney volume is the whole kidney (both kidneys together if both
  are in the scan). Tumor/cyst volumes are the lesion volumes only.
- `lesions[].center_xyz` is normalised 0..1 in each axis (D, H, W order),
  so the 3D viewer can place spheres without knowing the CT shape.
  One entry per class with voxels > 0 (v1: a single bounding lesion
  per class, not one entry per connected blob).

### Scores (only when `ground_truth` is given)

`scores` is `null` unless a ground-truth mask is uploaded. Then it is:

```json
{"dice": {"kidney": 0.97, "tumor": 0.82, "cyst": 0.55},
 "surface_dice_mm1": {"kidney": 0.93, "tumor": 0.78, "cyst": 0.50},
 "hd95_mm": {"kidney": 2.1, "tumor": 4.3, "cyst": 6.0},
 "note": "Dice is None for a class absent in both prediction and ground truth (no overlap to measure)."}
```

- Per class: **kidney, tumor, cyst only — no background**. Always report
  the three apart — the KiTS23 average hides the cyst score.
- In dummy mode `scores` stays `null` even when a ground truth is
  uploaded — scoring a test mask would fake a metric. Scores are
  real-mode only.
- Dice: overlap 0..1 per class, `null` for a class empty in both masks.
  It is never faked as 1.0 or 0.0.
- Surface Dice: edge overlap within 1 mm tolerance.
- HD95: 95th-percentile Hausdorff distance in mm (lower is better).

## Modes

- `dummy` (default, `KIDNEYSCAN_MODE=dummy`): no model file, no
  torch/nibabel. Mask is seeded by the uploaded file bytes. Response
  `is_dummy: true`, `model.name: "dummy-test-mask"`, `note` starts with
  `DUMMY / TEST`. The UI shows the DUMMY badge. Dummy output is NEVER
  a real prediction.
- `real` (`KIDNEYSCAN_MODE=real`): loads `backend/models_store/model.pt`
  (light 3D U-Net, PyTorch) via `model/inference.py` and runs real
  inference. Needs torch + nibabel + scipy. If the checkpoint is missing,
  or torch/nibabel is not installed, or inference fails, `/api/predict`
  returns HTTP 503. **Never fall back to dummy silently.**
  A normal input must return `tumor_present: false, cyst_present: false`
  and an empty lesion list.

## Errors — 400 / 413 / 503 only

```json
{
  "status": "error",
  "error": "human-readable message, e.g. 'file is not a .nii.gz'",
  "detail": "same message (kept so existing clients reading .detail keep working)",
  "warning": "Assistive tool only. This is not a diagnosis. A doctor must confirm the result."
}
```

| Code | Meaning |
|---|---|
| 400 | Bad input: file is not a `.nii.gz`, empty file, unreadable NIfTI, ground truth shape differs from the prediction, bad `mode` / `kind` value. |
| 413 | File too large (default max 512 MB, `KIDNEYSCAN_MAX_UPLOAD_BYTES`). |
| 503 | Real model not available or real inference failed: checkpoint missing, torch/nibabel/scipy not installed, or the model raised. Never a silent dummy fallback. |

There is no 500 and no 501 in this contract.

## Rules for the web worker

1. Show `warning` on every result screen. Never hide it.
2. Show `finding` exactly as returned. If it says no tumor/cyst, show
   the clean scan with no highlight — do not add one.
3. Do not invent or round scores beyond what the JSON gives.
4. Upload selects the INPUT only. The model decides the output.
