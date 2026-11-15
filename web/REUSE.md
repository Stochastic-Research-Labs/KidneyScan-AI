# KidneyScan AI — Reuse table

Rule (user, 15 Nov 2026): do NOT rewrite from scratch. Reuse efficiently.
Novelty stays clear: this demo shows kidney / tumor / cyst in 3D from real
model output (KiTS23, 3 labels), not a hard-coded highlight. The upload
selects INPUT only; the model decides the output.

## Reused patterns and code

| # | What we reused | Source | License | What we changed and why | How to cite |
|---|---|---|---|---|---|
| 1 | Next.js frontend on Vercel + FastAPI backend structure; same-origin `/backend/*` proxy to port 8000; CORS for localhost:3000 | NeuroScan AI, Wajid Ali / Hamza Mustafa, `neuroscan-sync/hamza-source/frontend/next.config.js` and `backend/app/main.py` | NeuroScan team code (own project, supervised by Saira Ishtiaq) — reuse inside the same team | Same proxy and CORS idea, rewritten as `next.config.ts` (Next 15, typed config). No NiiVue, no case store in v1 — result lives in sessionStorage | "Frontend/backend structure adapted from NeuroScan AI (Wajid Ali, Hamza Mustafa, supervised by Saira Ishtiaq)." |
| 2 | R3F Canvas + OrbitControls viewer pattern, lights, error fallback idea | NeuroScan `frontend/components/landing/BrainScene.tsx` (R3F Canvas, OrbitControls via drei, lights) | Same team code as above | Brain `.glb` model replaced by a schematic kidney (scaled sphere) — no kidney .glb exists in this repo. No scroll journey / GSAP. Two side-by-side panels instead of one landing scene. Lesion spheres are placed from the prediction `lesions[]` list, not hard-coded | Same citation as #1 |
| 3 | Two-panel comparison layout (left original / right predicted), legend, viewer controls (rotate, zoom, reset, fullscreen) | NeuroScan `frontend/components/viewer/ComparisonViewer.tsx` and `ViewerControls.tsx` | Same team code as above | NiiVue volumes replaced by R3F scenes (no NIfTI parsing in the browser in v1). Colours changed to KiTS23: kidney grey, tumor yellow/red, cyst blue. Added auto-rotate toggle and per-panel fullscreen | Same citation as #1 |
| 4 | Dark clinical theme (colours, glass panel, eyebrow, page shell) | NeuroScan `frontend/app/globals.css` and `tailwind.config.ts` | Same team code as above | Same palette values; simpler CSS (no Poppins local font, system font stack) so the build needs no font files | Same citation as #1 |
| 5 | Class-colour single source of truth, backend + frontend kept in sync by value | NeuroScan `frontend/lib/segmentation-colors.ts` and `backend/app/core/config.py` CLASS_INFO | Same team code as above | BraTS labels (1 NCR, 2 ED, 4 ET) replaced by KiTS23 labels (1 kidney, 2 tumor, 3 cyst) and new colours | Same citation as #1 |
| 6 | FastAPI `/api/health`, router layout, upload size guard, error codes (400/413/503/501) | NeuroScan `backend/app/main.py`, `app/api/routes/health.py`, `app/core/config.py` | Same team code as above | Single-file `backend/main.py` for v1 (no routers/case store). Dummy mode added for UI testing. Real mode returns 503 when the checkpoint is missing — never falls back to dummy silently | Same citation as #1 |
| 7 | nnU-Net / MONAI inference ideas — used as an API only | nnU-Net: https://github.com/MIC-DKFZ/nnUNet (Apache 2.0); MONAI: https://github.com/Project-MONAI/MONAI (Apache 2.0) | Apache 2.0 | No training code copied. Only the ideas are reused: 3D U-Net inference on a cropped CT, argmax to labels 0/1/2/3, per-class Dice / Surface Dice / HD95 scoring. The real model worker must implement `real_predict()` in `backend/main.py` against this contract | Isensee et al., Nature Methods 2021, DOI 10.1038/s41592-020-01008-z; Myronenko et al., arXiv:2310.04110 |
| 8 | KiTS23 data, labels, and challenge design | https://github.com/neheller/kits23, https://kits-challenge.org/kits23/ | CC BY-NC-SA 4.0 — free for research, not commercial | Data is not in this repo. Labels 0/1/2/3 and the 489 train / 110 test split are used as defined by the challenge | Heller et al., KiTS23 challenge; cite the challenge site and data repo |

## What is new here (not reused)

- Kidney + tumor + cyst together in one 3D demo (NeuroScan is brain tumor only).
- Cyst as a third class, with its own colour, volume, and count.
- Dummy mode that is seeded by the input file, plus a normal dummy sample
  that proves a normal case shows no highlight.
- API contract with a downsampled base64 mask (32³) and a per-lesion list,
  so the viewer renders real model output, not a fixed picture.
- Per-class Surface Dice + HD95 in the contract (Dice alone hides bad edges
  and the cyst problem — see `../docs/literature-review.md`).

## What we did NOT copy

- No nnU-Net / MONAI training code. No BraTS code or weights.
- No NeuroScan brain model (.glb), no patient data, no screenshots.
- No hard-coded tumor/cyst position in real mode — the viewer only draws
  what the prediction JSON lists.
