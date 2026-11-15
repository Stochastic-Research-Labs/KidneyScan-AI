# KidneyScan AI — Push Checklist

Prepared: 2026-11-15. Do NOT push until the two BLOCKED items below are done.
Target repo (updated 2026-11-15 23:37 PKT): **Stochastic-Research-Labs/KidneyScan-AI** — Stochastic Research org, same org as NeuroScan-AI. NOT personal `wajidcodes/KidneyScan-AI`. Do not push to the personal account.

## What will be pushed

- 56 files, about 357 KB total (480 KB on disk, checked 2026-11-15).
  Note: another worker is still adding files to this folder, so re-check the
  count with `git status` before you push.
- Code + docs + small configs only.
- Biggest file: `web/frontend/package-lock.json` (96 KB). No file over 1 MB.

| Folder | Files | What it is |
|---|---|---|
| `data-prep/` | 6 | Download, check, preprocess, splits, synthetic tests |
| `docs/` | 7 | Literature review, deploy plan, help-for-others, demo script, LinkedIn draft, robustness template |
| `model/` | 12 | Light 3D U-Net, train, evaluate, inference, losses, Kaggle notebook, ablation plan, robustness |
| `web/backend/` | 3 | FastAPI `main.py`, `requirements.txt`, `.env.example` |
| `web/frontend/` | 14 | Next.js app, viewer components, config files |
| `web/docs/` + `web/` | 3 | API contract, READMEs |
| Root | 2 | `.gitignore`, this checklist |

## What is EXCLUDED (about 850 MB bloat — confirmed)

| Path | Size | Why excluded |
|---|---|---|
| `web/frontend/node_modules/` | 560 MB | Rebuilt with `npm install` |
| `web/backend/.venv/` | 126 MB | Rebuilt with `python -m venv` |
| `web/frontend/.next/` | 118 MB | Rebuilt with `npm run build` |
| `.venv-phase1-test/` | 88 MB and growing | Test venv from another worker, also ignored (`.venv-*/`) |
| `__pycache__/`, `*.pyc` (15 files in code folders) | small | Python cache |
| `*.nii`, `*.nii.gz`, `data/`, `*.pt`, `*.pth`, `.env`, `*.log` | none present | No such files exist yet — the rules are in `.gitignore` so they can never slip in |

Total folder on disk is ~857 MB and still growing (another worker is active).
After the ignore rules, the push is only ~357 KB.

## Secrets check

- Grepped all code/docs for `password`, `token`, `key`, `secret`, private keys, `sk-`, `ghp_` — **none found**.
- Only `.env` file is `web/backend/.env.example` — it has dummy comments only, no real values. Safe to push.
- Never add a real `.env` — it is ignored by `.gitignore`.

## Steps to create the repo and push

Run these in `~/workspace/kidneyscan-ai/` on the machine that will push.
Do it once, in order.

```bash
# 1. Add a root README.md first (see BLOCKED below), then:
cd ~/workspace/kidneyscan-ai

# 2. Create the repo on GitHub first (github.com/new):
#    Owner: Stochastic-Research-Labs (the Stochastic Research org, NOT personal wajidcodes)
#    Name: KidneyScan-AI — Public — NO README, NO .gitignore, NO license added there.
#    Full repo: Stochastic-Research-Labs/KidneyScan-AI — same org as Stochastic-Research-Labs/NeuroScan-AI.
#    Needs org permission: only an org owner/member with repo-create rights can create it.
#    If Wajid cannot create it, ask the org owner (same owner as NeuroScan-AI) to create it or give him access first.

# 3. Init and push
git init
git add .
git status          # CHECK: only the 56 files above. If you see node_modules, .venv, .next, .env or any .pt/.nii file — STOP.
git commit -m "Initial commit: KidneyScan AI — lightweight 3D kidney tumor segmentation (KiTS23)"
git branch -M main
git remote add origin https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git
git push -u origin main
```

After the push, check on github.com/Stochastic-Research-Labs/KidneyScan-AI that:
- The file count looks right and no `node_modules`, `.venv`, or `.next` folder is there.
- The README shows on the repo home page.

## What NOT to commit (ever)

- `node_modules/`, `.venv/`, `venv/`, `.next/`, `dist/`, `build/`
- `__pycache__/`, `*.pyc`, `*.log`
- `data/` and any `*.nii` / `*.nii.gz` scan files (KiTS23 data is 94.8 GB)
- Any `*.pt` / `*.pth` checkpoint or weight file, `checkpoints/`
- Any real `.env` file, password, token, or API key
- The KiTS23 dataset itself — users download it with `data-prep/download.py`

## License note

- **Code: MIT.** No `LICENSE` file exists yet — add an MIT `LICENSE` file at the repo root before the first push (see BLOCKED).
- **Data: CC BY-NC-SA 4.0.** KiTS23 data is NOT in this repo and must never be added. It is covered by CC BY-NC-SA 4.0, non-commercial use only. Link to it instead: https://kits-challenge.org/kits23/ and https://github.com/neheller/kits23
- Reused code is recorded in `model/REUSE.md` and `web/REUSE.md` — keep those files, they show source + license + what changed.

## README note

- There is **no root `README.md` yet**. Folder READMEs exist (`docs/README.md`, `model/README.md`, `web/README.md`, `data-prep/README.md`), but GitHub shows only the root README on the repo home page.
- Before pushing, add a short root `README.md`: what KidneyScan AI is, the 6 gaps it covers, how to run data-prep / model / web, link to KiTS23 data, license lines (code MIT, data CC BY-NC-SA 4.0), and "learning / research only, not medical diagnosis".

## BLOCKED — do these before pushing

1. **Add root `README.md`** (see note above).
2. **Add MIT `LICENSE` file** at the repo root.
3. Nothing else blocks the push. No secrets found, no large files, no data files in the folder.
