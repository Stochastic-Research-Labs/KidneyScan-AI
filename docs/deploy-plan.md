# KidneyScan AI — Deploy Plan

Date: 15 Nov 2026.
Status: **NOT deployed. NOT pushed. NOT trained.**
Pattern: same as NeuroScan AI (live at https://neuroscan-ai-research.vercel.app) —
Next.js frontend on Vercel + FastAPI backend on Modal, 1 CPU, model baked in.

## 1. What is ready, what is not

Ready now:
- Frontend builds locally: `web/frontend`, Next.js 15.5.0, React 19.3, `npm run build` = `next build`.
- Backend runs locally in dummy mode: `web/backend/main.py`, FastAPI, port 8000.
- Dummy mode needs no model, no torch, no HF token.

NOT ready — do not deploy real mode until these are done:
- No trained checkpoint. `web/backend/models_store/` is empty.
- `real_predict()` in `web/backend/main.py` is wired to `model/inference.py` since contract v1.1 (15 Nov 2026), but there is no checkpoint and no torch/nibabel installed yet — real mode returns 503 until those exist. There is no 501 any more.
- No Modal file exists yet in this repo (no `modal_app.py`). One must be written first.
- No GitHub repo for KidneyScan yet. No Vercel project yet. No Modal app yet.
- CORS in `main.py` allows only `http://localhost:3000`. The Vercel domain must be added before deploy, or the live frontend cannot call Modal direct.

## 2. API contract — read this first

The endpoint the code really uses is:

```
POST /api/predict
```

- Multipart fields: `file` (.nii.gz, required), `ground_truth` (.nii.gz, optional), `mode` form field (`dummy` or `real`, optional).
- Also: `GET /api/health`, `GET /api/demo/sample?kind=tumor|both|normal`, `GET /`, `GET /api/predict/{case_id}/mask` (full-res mask `.nii.gz` download). `POST /predict` is an alias of `POST /api/predict`.
- Response: `PredictionResult` in `API_CONTRACT.md` (canonical, at the repo root) — `status, case_id, mode, is_dummy, model{}, input, finding, findings, volumes_cm3, lesions[], mask{file}, mask_base64` (32x32x32 preview), `scores` (kidney/tumor/cyst only, no background), `warning`.
- Frontend call: `frontend/lib/api.ts` uses `NEXT_PUBLIC_API_BASE_URL ?? "/backend"`, then fetches `${API_BASE}/api/predict`.

Contract fixed 15 Nov 2026 (v1.1): the old model contract (`POST /predict`, field `file`, mask file, `finding` string) and the old web contract (`POST /api/predict`, field `ct`, `mask_base64`, `findings` object) were merged into one canonical contract at `kidneyscan-ai/API_CONTRACT.md`. Both old files are now pointers to it. Deploy against the canonical one: `POST /api/predict`, field `file`.

Local proxy note: `frontend/next.config.ts` rewrites `/backend/*` to `http://127.0.0.1:8000/*`.
That is local-only. On Vercel it does nothing useful. In production the frontend must call Modal direct through `NEXT_PUBLIC_API_BASE_URL` (see §4).

## 3. Pre-deploy fixes (code, before any deploy)

1. Model worker wires `real_predict()` in `web/backend/main.py` (load NIfTI, run `models_store/model.pt`, argmax to labels 0/1/2/3) and trains the checkpoint. Real mode is blocked until then.
2. Add the Vercel domain to `allow_origins` in `main.py` (keep localhost for dev).
3. Uncomment real-mode deps in `web/backend/requirements.txt`: `torch`, `nibabel`, `scipy`. Torch must be CPU-only for Modal 1 CPU.
4. Write the Modal app file (see §5). Do not put any token or secret in it.
5. Test locally first: `KIDNEYSCAN_MODE=real` + real checkpoint, one real CT, check `/api/health` says `model_loaded: true`.

## 4. Part 1 — Vercel frontend

Repo root for the Vercel project: `web/frontend` (not the repo root).

Steps:
1. Push the code to GitHub first (see §8 — user browser step).
2. In Vercel: New Project → Import the GitHub repo → set Root Directory to `web/frontend`.
3. Framework preset: Next.js (auto-detected).
4. Build command: `npm run build` (default `next build`). Install command: `npm install`. Output: default.
5. Node version: 20+ (frontend needs Node 20+).
6. Env var (Production + Preview):
   - `NEXT_PUBLIC_API_BASE_URL` = Modal backend URL, e.g. `https://wajidcodes--kidneyscan-backend-serve.modal.run` (exact URL comes from the Modal deploy in §5 — deploy Modal FIRST, then paste its URL here). No trailing slash.
   - No other env var is needed for the frontend. Do NOT set a dummy/real flag here — mode is picked on the Upload page and sent as the `mode` form field.
7. Deploy. Then redeploy after every change to this env var (Next.js bakes `NEXT_PUBLIC_*` vars into the build).

Check: open the Vercel URL. Home page loads. Upload page shows. Health check in the browser console / network tab calls `<Modal URL>/api/health`.

## 5. Part 2 — Modal backend

Same pattern as NeuroScan: FastAPI on Modal, 1 CPU, model baked into the image.

Steps:
1. Write `web/backend/modal_app.py` (new file, not done yet). Sketch:
   - Modal image: Python 3.12, `pip install -r requirements.txt` (CPU torch).
   - Copy into the image: `main.py`, `requirements.txt`, and `models_store/model.pt` (baked in — see below).
   - App: `modal.App("kidneyscan-ai")`, `@app.function(cpu=1, scaledown_window=300)` serving FastAPI with `modal.asgi_app()`.
   - Env in the function: `KIDNEYSCAN_MODE=real` (or `dummy` for a test deploy).
   - No HF token, no download at runtime. The checkpoint is already in the image.
2. Deploy from a terminal with Modal signed in: `modal deploy web/backend/modal_app.py` (exact command depends on the file layout in step 1).
3. Copy the serve URL Modal prints. That URL is the `NEXT_PUBLIC_API_BASE_URL` for Vercel (§4).
4. Cold start: first request after idle can be slow (model load on 1 CPU). That is normal. Keep the model light (1.68M params, light 3D U-Net) so load stays fast.

Why bake the model in:
- No HF token at runtime, no download fail, no 94.8 GB data needed.
- Same as NeuroScan: checkpoint travels with the image.
- Cost: image is bigger, but deploy is simple and repeatable.

Limits to know:
- 1 CPU only. One CT at a time is fine. Do not promise fast batch use.
- Max upload is 512 MB by default (`KIDNEYSCAN_MAX_UPLOAD_BYTES`). A normal KiTS23 CT is well under this.
- Modal sleeps when idle (scaledown). First predict after sleep takes longer.

## 6. Env vars — full list

Frontend (Vercel, Production):

| Var | Value | Notes |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Modal URL, no trailing slash | Required in production. If missing, frontend falls back to `/backend`, which does not work on Vercel. |

Backend (Modal, set in the Modal function, not in Vercel):

| Var | Value | Notes |
|---|---|---|
| `KIDNEYSCAN_MODE` | `real` (live) / `dummy` (test) | Default is `dummy` if unset. |
| `KIDNEYSCAN_MAX_UPLOAD_BYTES` | `536870912` (512 MB, default) | Only change if a real CT is bigger. |

No secrets needed:
- No HF token at runtime (model baked in).
- No database, no email, no API key. Do not add any.

Local dev only (do not use in production):
- Backend `.env` example is `web/backend/.env.example` (`KIDNEYSCAN_MODE`, `KIDNEYSCAN_MAX_UPLOAD_BYTES`).
- Local frontend uses `/backend` proxy to `127.0.0.1:8000`. No env var needed locally.

## 7. Dummy vs real mode switch

| Mode | How to turn it on | What the user sees | If it fails |
|---|---|---|---|
| Dummy (default) | `KIDNEYSCAN_MODE=dummy`, or pick "Dummy" on the Upload page, or use `/api/demo/sample` | Yellow **DUMMY / TEST** badge on viewer + results. Mask is a seeded test shape, not a real prediction. "Dummy: normal" shows kidney only, no highlight. | Dummy never needs a model. If dummy fails, the backend itself is down — check Modal logs. |
| Real | `KIDNEYSCAN_MODE=real` on Modal + checkpoint baked in + `real_predict()` wired, pick "Real model" on the Upload page | No dummy badge. Real volumes, real lesions from the model. Normal input shows no tumor/cyst highlight. | If `model.pt` is missing, `POST /api/predict` returns **503**. It never falls back to dummy silently. If `real_predict()` is still a stub, it returns **501**. Both are correct — fix the model, do not hide the error. |

Rule: the upload selects INPUT only. The model decides the output. Never hard-code a tumor position in real mode.

Safe order: deploy dummy on Modal + Vercel first to prove the full path works, then swap Modal to real mode and redeploy once the checkpoint exists.

## 8. Domain name suggestion

Suggested, following the NeuroScan pattern (no personal name, team project):
- First choice: `kidneyscan-ai-research.vercel.app`
- Fallback if taken: `kidneyscan-ai.vercel.app`

Set it in Vercel → Project → Settings → Domains after the first deploy. Check it is free on the Vercel domains page — do not assume. The clean URL to share is the `.vercel.app` URL above, same as NeuroScan's `https://neuroscan-ai-research.vercel.app`.

## 9. Test checklist (run after deploy)

Do these in order. Stop at the first fail and fix it before moving on.

- [ ] 1. `GET <Modal URL>/api/health` returns `{"status": "ok", "mode": "real" (or "dummy"), "checkpoint_present": true}` in real mode.
- [ ] 2. Open the Vercel URL. Home page loads, no blank page, no console error about the API.
- [ ] 3. Upload page → "Dummy: tumor + cyst" (no file). Viewer shows kidney + tumor + cyst, DUMMY badge is visible.
- [ ] 4. Upload page → "Dummy: normal (no highlight)". Viewer shows kidney only, no tumor/cyst highlight.
- [ ] 5. Upload a real `.nii.gz` CT in dummy mode → predict returns 200, `is_dummy: true`, volumes shown.
- [ ] 6. Same upload in real mode → predict returns 200, `is_dummy: false`, no dummy badge. If no checkpoint, it must return 503 (not a fake result).
- [ ] 7. Upload a real CT + ground-truth mask in real mode → results page shows Dice / Surface Dice / HD95 per class (kidney, tumor, cyst apart).
- [ ] 8. Upload a normal CT in real mode → result says no tumor, no cyst, viewer shows no highlight.
- [ ] 9. 3D view: drag to rotate 360°, scroll to zoom, reset + fullscreen buttons work on both panels (original left, prediction right).
- [ ] 10. Bad file test: upload a `.jpg` → 400 error, clear message. Empty file → 400.
- [ ] 11. Cold-start test: wait 10 minutes idle, then predict again. It is slower but must still return 200.
- [ ] 12. Check the warning is on every result screen: "Assistive only. A doctor must confirm. Not a diagnosis."

## 10. Costs (free tiers, check at deploy time)

- Vercel Hobby: free. Frontend hosting, bandwidth and builds are inside the free tier for a demo like this. No card needed for Hobby.
- Modal: free tier with monthly free compute credit (check the current amount on Modal's pricing page at deploy time — it changes). 1 CPU, light model, demo traffic should stay inside free credit. If traffic grows, Modal bills per second of CPU use — set a spending limit in Modal if the account offers one.
- GitHub: free for a public repo.
- Total expected cost for a demo: $0 on free tiers. No paid plan is needed to launch.

## 11. What needs a USER BROWSER step

A subagent cannot do these. The user (or parent agent with the browser) must do them:

1. **GitHub repo (ORG, not personal)** — create `Stochastic-Research-Labs/KidneyScan-AI` (public) in the Stochastic Research org, same org as `Stochastic-Research-Labs/NeuroScan-AI`, and push `kidneyscan-ai/` code to it. Do NOT create `wajidcodes/KidneyScan-AI`. Needed before Vercel can import it. Needs org permission: only an org owner/member with repo-create rights can create it — if Wajid cannot, the org owner must create it or grant access first.
2. **Vercel sign-in** — sign in to Vercel, New Project → Import the GitHub repo, set Root Directory `web/frontend`, add `NEXT_PUBLIC_API_BASE_URL`, deploy, add the domain in §8.
3. **Modal sign-in** — sign in to Modal in the terminal/browser, create/confirm the account, run `modal deploy`. The Modal serve URL from this step is needed for the Vercel env var.
4. **Domain check** — confirm `kidneyscan-ai-research.vercel.app` is free in Vercel, or pick the fallback in §8.

No other browser step is needed. No payment, no HF token, no new account besides GitHub / Vercel / Modal (all already used for NeuroScan).

## 12. If deploy fails

- Frontend build fails on Vercel: run `npm run build` locally in `web/frontend` first and fix the error there. Do not guess on Vercel.
- Frontend loads but predict fails: open the browser network tab. If it calls `/backend/api/predict`, `NEXT_PUBLIC_API_BASE_URL` was not set (or not redeployed after setting it).
- CORS error in the browser: the Vercel domain is missing from `allow_origins` in `main.py` (§3, fix 2).
- 503 on real predict: checkpoint missing in the Modal image — rebuild the image with `model.pt` baked in. (No 501 exists any more: real inference is wired since contract v1.1; a missing checkpoint/torch is a 503.)
- Modal cold start is slow: normal on 1 CPU. Only worry if it times out — then check Modal logs for out-of-memory.

## Sources used for this plan

- `web/README.md`, `web/REUSE.md`
- `web/frontend/package.json` (Next 15.5.0), `web/frontend/next.config.ts` (`/backend` → `127.0.0.1:8000`), `web/frontend/lib/api.ts` (`NEXT_PUBLIC_API_BASE_URL ?? "/backend"`)
- `web/backend/main.py` (modes, 503, 512 MB limit), `web/backend/requirements.txt`, `web/backend/.env.example`, `web/backend/models_store/` (empty)
- `API_CONTRACT.md` (canonical contract v1.1, repo root); `web/docs/API_CONTRACT.md` and `model/API_CONTRACT.md` are pointers to it
- NeuroScan live pattern from memory: `https://neuroscan-ai-research.vercel.app`, Next.js on Vercel + FastAPI on Modal, 1 CPU, model baked in.
