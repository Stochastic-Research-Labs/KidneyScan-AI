"""KidneyScan AI backend — FastAPI.

Implements the canonical API contract v1.1: see
``../../API_CONTRACT.md`` (``web/docs/API_CONTRACT.md`` is only a
pointer to it).

Modes:
- dummy (default, KIDNEYSCAN_MODE=dummy): no trained model needed. Generates a
  deterministic test mask from the uploaded file's bytes (seeded, so a different
  input gives a different output) and returns it labelled mode="dummy".
  Dummy output is NEVER a real prediction. The UI must show the DUMMY badge.
- real (KIDNEYSCAN_MODE=real): loads models_store/model.pt (light 3D U-Net,
  PyTorch) and runs real inference through model/inference.py.
  Needs torch + nibabel + scipy installed.
  If the checkpoint is missing, /api/predict returns 503 — it never falls
  back to dummy silently.

Real-AI rule: the upload selects INPUT only. The model decides the output.
A normal input returns no tumor and no cyst, so the viewer shows no highlight.

Errors are 400 / 413 / 503 only (no 500, no 501). Error body:
{"status": "error", "error": ..., "detail": ..., "warning": ...}.
"""
import base64
import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

BACKEND_ROOT = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_ROOT.parent.parent / "model"  # kidneyscan-ai/model
MODELS_DIR = BACKEND_ROOT / "models_store"
MASKS_DIR = BACKEND_ROOT / "masks_store"
MASKS_DIR.mkdir(exist_ok=True)
CHECKPOINT = MODELS_DIR / "model.pt"
MODE = os.getenv("KIDNEYSCAN_MODE", "dummy").lower()
MAX_UPLOAD_BYTES = int(os.getenv("KIDNEYSCAN_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))
APP_VERSION = "0.1.0"

ASSISTIVE_WARNING = (
    "Assistive tool only. This is not a diagnosis. "
    "A doctor must confirm the result."
)

# KiTS23 labels. Must stay in sync with frontend lib/segmentation-colors.ts
# and lib/api.ts CLASS_COLORS. 0 background, 1 kidney, 2 tumor, 3 cyst.
CLASS_INFO = {
    "kidney": {"label_value": 1, "name": "Kidney", "color": "#9aa7b0"},
    "tumor": {"label_value": 2, "name": "Tumor", "color": "#facc15"},
    "cyst": {"label_value": 3, "name": "Cyst", "color": "#38bdf8"},
}
CLASS_NAMES = ["background", "kidney", "tumor", "cyst"]
# Downsampled mask grid returned in every /api/predict response (D, H, W).
# This is a viewer preview only — the full mask is the .nii.gz file.
MASK_GRID = 32
# Dummy mask is built on this grid; spacing is stated, not measured from a header.
DUMMY_SPACING_MM = (1.0, 1.0, 1.0)

app = FastAPI(
    title="KidneyScan AI backend",
    description="Kidney tumor + cyst segmentation in 3D CT (KiTS23). Assistive only.",
    version=APP_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- errors
@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Canonical error body (API_CONTRACT.md): status/error/detail/warning.
    `detail` is kept so existing clients reading .detail keep working."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": detail, "detail": detail,
                 "warning": ASSISTIVE_WARNING},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """A missing/invalid form field (e.g. no `file` uploaded) is a 400
    bad input in this contract, not FastAPI's default 422."""
    return JSONResponse(
        status_code=400,
        content={"status": "error",
                 "error": "Bad request: upload one CT file in the 'file' field (.nii.gz).",
                 "detail": "Bad request: upload one CT file in the 'file' field (.nii.gz).",
                 "warning": ASSISTIVE_WARNING},
    )


# ---------------------------------------------------------------- helpers
def _safe_case_id(raw: str) -> str:
    """File-safe case id for the mask store (no slashes, no '..')."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", raw).strip("._") or "case"
    return safe[:100]


def finding_from_counts(counts: dict) -> str:
    """Canonical `finding` string (API_CONTRACT.md), from voxel counts."""
    if counts["tumor"] == 0 and counts["cyst"] == 0:
        return "no tumor or cyst detected"
    if counts["tumor"] > 0 and counts["cyst"] == 0:
        return "tumor detected, no cyst detected"
    if counts["tumor"] == 0 and counts["cyst"] > 0:
        return "cyst detected, no tumor detected"
    return "tumor and cyst detected"


def _lesions_from_mask(mask: np.ndarray, label: int, spacing: tuple[float, float, float]) -> list[dict]:
    """One entry per class with voxels > 0 (v1: a single bounding lesion per
    class, not one entry per connected blob). Centre is normalised 0..1 in
    each axis (D, H, W order); radius in mm is derived from the volume."""
    vox = mask == label
    count = int(vox.sum())
    if count == 0:
        return []
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    volume_cm3 = count * voxel_cm3
    coords = np.argwhere(vox)
    centre = (coords.mean(axis=0) / np.array(mask.shape)).tolist()
    radius_mm = float((3 * volume_cm3 * 1000 / (4 * np.pi)) ** (1 / 3))
    return [{
        "class": "tumor" if label == 2 else "cyst",
        "center_xyz": [round(float(c), 3) for c in centre],
        "radius_mm": round(radius_mm, 1),
        "volume_cm3": round(volume_cm3, 2),
    }]


def _save_mask_file(mask: np.ndarray, spacing, affine, case_id: str) -> str | None:
    """Save the full-resolution mask as .nii.gz for the download endpoint.
    Returns the download URL, or None when nibabel is not installed
    (dummy mode must work without nibabel)."""
    try:
        import nibabel as nib
    except ImportError:
        return None
    if affine is None:
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    path = MASKS_DIR / f"{_safe_case_id(case_id)}.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))
    return f"/api/predict/{_safe_case_id(case_id)}/mask"


def build_response(mask: np.ndarray, spacing, mode: str, case_id: str,
                   model_meta: dict, input_file: str,
                   scores: dict | None = None, note: str = "",
                   inference_seconds: float = 0.0,
                   affine=None) -> dict:
    """Assemble the canonical PredictionResult (API_CONTRACT.md v1.1).

    Core fields match what model/inference.py predict() returns; the
    backend adds the web-only viewer fields (lesions, mask_grid,
    mask_base64, labels) and the mask .nii.gz download URL.
    """
    t0 = time.time()
    spacing = tuple(float(s) for s in spacing)
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    counts = {k: int((mask == v["label_value"]).sum()) for k, v in CLASS_INFO.items()}
    volumes = {k: round(c * voxel_cm3, 2) for k, c in counts.items()}
    lesions = _lesions_from_mask(mask, 2, spacing) + _lesions_from_mask(mask, 3, spacing)
    mask_file_url = _save_mask_file(mask, spacing, affine, case_id)
    # Downsampled mask (base64, uint8, row-major D,H,W): viewer preview only.
    small = mask
    if mask.shape != (MASK_GRID, MASK_GRID, MASK_GRID):
        idx = tuple(np.linspace(0, s - 1, MASK_GRID).astype(int) for s in mask.shape)
        small = mask[np.ix_(*idx)]
    return {
        "status": "ok",
        "case_id": case_id,
        "mode": mode,
        "is_dummy": mode == "dummy",
        "model": model_meta,
        "note": note,
        "input": {
            "file": input_file,
            "shape": list(mask.shape),
            "spacing_mm": [round(s, 3) for s in spacing],
        },
        "shape": list(mask.shape),
        "spacing_mm": [round(s, 3) for s in spacing],
        "finding": finding_from_counts(counts),
        "findings": {
            "tumor_present": counts["tumor"] > 0,
            "cyst_present": counts["cyst"] > 0,
            "cyst_count": len(_lesions_from_mask(mask, 3, spacing)),
            "tumor_volume_cm3": volumes["tumor"],
            "cyst_volume_cm3": volumes["cyst"],
        },
        "volumes_cm3": volumes,
        "voxel_counts": counts,
        "lesions": lesions,
        "mask": {
            "encoding": "nifti-gz file, same shape/orientation as input; "
                        "values 0 background, 1 kidney, 2 tumor, 3 cyst",
            "file": mask_file_url,
            "shape": list(mask.shape),
        },
        "mask_grid": MASK_GRID,
        "mask_base64": base64.b64encode(small.astype(np.uint8).tobytes()).decode(),
        "labels": {"0": "background", "1": "kidney", "2": "tumor", "3": "cyst"},
        "scores": scores,  # Dice / Surface Dice / HD95 per class, only when GT given
        "inference_seconds": round(inference_seconds if inference_seconds else time.time() - t0, 2),
        "warning": ASSISTIVE_WARNING,
    }


def _dummy_model_meta() -> dict:
    return {"name": "dummy-test-mask", "checkpoint": None,
            "base_channels": 48, "num_classes": 4, "classes": CLASS_NAMES}


# ---------------------------------------------------------------- dummy model
def _ellipsoid(shape, centre, radii) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return (((zz - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2
            + ((xx - centre[2]) / radii[2]) ** 2) <= 1


def dummy_mask(data: bytes, kind: str = "tumor") -> np.ndarray:
    """Deterministic TEST mask, seeded by the input bytes. Different input ->
    different blobs. kind=normal returns kidney only (no tumor, no cyst),
    which is how the viewer proves a normal case shows no highlight."""
    seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    # 128^3 grid at 1 mm spacing: kidney ~140 cm3, tumor ~7 cm3, cyst ~2 cm3 —
    # plausible sizes for a test mask. build_response() downsamples to 32^3
    # for the base64 mask. These are still TEST values, never real volumes.
    shape = (128, 128, 128)
    mask = np.zeros(shape, dtype=np.uint8)
    # Kidney: one bean-ish ellipsoid, slightly off-centre (seeded).
    kc = (64 + int(rng.integers(-8, 9)), 64, 64)
    mask[_ellipsoid(shape, kc, (45, 30, 25))] = 1
    if kind == "normal":
        return mask
    # Tumor: small blob on the kidney surface.
    tc = (kc[0] + int(rng.integers(-16, 17)), kc[1] + int(rng.integers(-8, 9)), kc[2] + 20)
    mask[_ellipsoid(shape, tc, (12, 12, 12))] = 2
    if kind in ("tumor", "both"):
        # Cyst: smaller blue blob elsewhere on the kidney.
        cc = (kc[0] - 16, kc[1] + 12, kc[2] - 8)
        region = _ellipsoid(shape, cc, (8, 8, 8)) & (mask == 1)
        mask[region] = 3
    return mask


# ---------------------------------------------------------------- real model
def real_predict(data: bytes, gt_data: bytes | None = None):
    """Real inference via the model worker (model/inference.py predict()).

    Returns (model_result, mask, affine). Raises 503 if the checkpoint is
    missing, torch/nibabel is not installed, or inference fails — never
    falls back to dummy. Raises 400 for an unreadable input file.
    """
    if not CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail="Real model not available: models_store/model.pt is missing. "
                   "Train the light 3D U-Net first, or use dummy mode for UI testing.",
        )
    if not MODEL_DIR.exists():
        raise HTTPException(
            status_code=503,
            detail="Real model not available: model worker code (kidneyscan-ai/model) not found.",
        )
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    try:
        import inference as model_inference
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Real mode needs torch + nibabel installed: {exc}",
        )
    with tempfile.TemporaryDirectory(prefix="kidneyscan-") as tmp:
        ct_path = Path(tmp) / "input.nii.gz"
        ct_path.write_bytes(data)
        gt_path = None
        if gt_data is not None:
            gt_path = Path(tmp) / "ground_truth.nii.gz"
            gt_path.write_bytes(gt_data)
        try:
            result, mask, affine = model_inference.predict(
                ct_path, CHECKPOINT, ground_truth_path=gt_path)
        except ImportError as exc:
            # torch / nibabel missing inside the model worker.
            raise HTTPException(
                status_code=503,
                detail=f"Real mode needs torch + nibabel installed: {exc}",
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"Real model failed: {exc}")
        except ValueError as exc:
            # Bad input: unreadable NIfTI, ground truth shape mismatch, etc.
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:  # model failed (e.g. out of memory)
            raise HTTPException(status_code=503, detail=f"Real model failed: {exc}")
        return result, mask, affine


# ---------------------------------------------------------------- routes
@app.get("/")
def root() -> dict:
    return {"name": "KidneyScan AI backend", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "mode": MODE,
        "model_loaded": CHECKPOINT.exists() if MODE == "real" else False,
        "checkpoint_present": CHECKPOINT.exists(),
    }


@app.get("/api/demo/sample")
def demo_sample(kind: str = "tumor") -> dict:
    """Test sample without uploading a file. kind=tumor (default), both, or normal.
    Always mode=dummy. Use kind=normal to check that a normal case shows no highlight."""
    if kind not in ("tumor", "both", "normal"):
        raise HTTPException(status_code=400, detail="kind must be tumor, both, or normal")
    data = f"kidneyscan-demo-{kind}".encode()
    mask = dummy_mask(data, kind)
    return build_response(mask, DUMMY_SPACING_MM, "dummy", f"demo-{kind}",
                          _dummy_model_meta(), f"demo-{kind}.nii.gz",
                          note=f"DUMMY / TEST sample ({kind}). Not a real prediction.")


@app.post("/api/predict")
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    ground_truth: UploadFile | None = File(default=None),
    mode: str = Form(default=""),
) -> dict:
    """Upload one CT (.nii.gz). Returns the canonical PredictionResult
    (../../API_CONTRACT.md). Optional ground_truth (.nii.gz, same shape)
    adds per-class Dice / Surface Dice / HD95 (kidney/tumor/cyst only).
    `/predict` is an alias of `/api/predict` (same handler)."""
    use_mode = (mode or MODE).lower()
    if use_mode not in ("dummy", "real"):
        raise HTTPException(status_code=400, detail="mode must be dummy or real")
    name = file.filename or "case.nii.gz"
    if not name.endswith(".nii.gz"):
        raise HTTPException(status_code=400, detail="Upload a CT file ending in .nii.gz")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 512 MB by default)")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    case_id = _safe_case_id(Path(name).name.replace(".nii.gz", "") or "case")

    gt_data = None
    if ground_truth is not None:
        gt_name = ground_truth.filename or "ground_truth.nii.gz"
        if not gt_name.endswith(".nii.gz"):
            raise HTTPException(status_code=400,
                                detail="Ground truth must be a .nii.gz file")
        gt_data = await ground_truth.read()
        if len(gt_data) == 0:
            raise HTTPException(status_code=400, detail="Empty ground-truth file")

    if use_mode == "dummy":
        # Seeded by file bytes: same file -> same dummy mask; new file -> new mask.
        mask = dummy_mask(data, "both")
        return build_response(mask, DUMMY_SPACING_MM, "dummy", case_id,
                              _dummy_model_meta(), name,
                              note="DUMMY / TEST prediction. Generated for UI testing only. "
                                   "Not a real model result — switch to real mode for real inference.")

    # Real mode: model worker decides. Never falls back to dummy.
    result, mask, affine = real_predict(data, gt_data)
    return build_response(mask, result["spacing_mm"], "real", case_id,
                          result["model"], name, scores=result["scores"],
                          inference_seconds=result["inference_seconds"],
                          affine=affine)


@app.get("/api/predict/{case_id}/mask")
@app.get("/predict/{case_id}/mask")
def download_mask(case_id: str):
    """Full-resolution predicted mask as a .nii.gz file (API_CONTRACT.md).
    404 when no mask was saved for this case (e.g. dummy mode without
    nibabel installed, or an unknown case id)."""
    path = MASKS_DIR / f"{_safe_case_id(case_id)}.nii.gz"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No saved mask for case '{case_id}'")
    return FileResponse(path, media_type="application/gzip",
                        filename=f"{_safe_case_id(case_id)}_pred.nii.gz")
