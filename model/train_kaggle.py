"""
KidneyScan AI — Kaggle training entry point (train_kaggle.py)
=============================================================
Ready-to-run on Kaggle GPU (PyTorch preinstalled there).

What it does:
  1. Clones this model folder from GitHub (or uses the attached copy).
  2. Downloads KiTS23 data (Lite 40.9 GB) from Hugging Face.
  3. Runs train.py Phase 1 (20 cases, 30 epochs) or Phase 2 (400/89).

Kaggle setup (do this once in the notebook / Kaggle UI):
  - GPU accelerator ON (T4/P100, free, 30 h/week).
  - Internet ON (needed for the Hugging Face download + GitHub clone).
  - Set the Kaggle secret or edit the variables below for the data path.

Run it:
    !python train_kaggle.py --phase phase1
    !python train_kaggle.py --phase phase2 --epochs 80

Checkpoints are written to ./checkpoints/ — download them from the
Kaggle output tab before the session ends, or save to a Kaggle Dataset.

NO full training run happens on this VM. This file only prepares the
run; the GPU work happens on Kaggle.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- edit these for your Kaggle setup -------------------------------------
GITHUB_REPO = "https://github.com/Stochastic-Research-Labs/KidneyScan-AI.git"
# Hugging Face KiTS23 Lite (40.9 GB). Full official set is 94.8 GB
# (neheller/KiTS-Challenge-Imaging) — do NOT use it here, it is the
# full download, not Lite. train.py reads the Lite Images/Masks layout.
HF_DATASET = "YongchengYAO/KiTS23-Lite"
# --------------------------------------------------------------------------


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="KidneyScan Kaggle runner")
    ap.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--data-dir", type=Path, default=Path("data/kits23"),
                    help="Folder holding case_XXXXX subfolders")
    ap.add_argument("--model-dir", type=Path, default=Path("kidneyscan-model"),
                    help="Folder with light_unet3d.py, train.py, ...")
    ap.add_argument("--skip-download", action="store_true",
                    help="Data already present, do not download")
    ap.add_argument("--skip-deps", action="store_true",
                    help="Do not pip install nibabel/scipy")
    args = ap.parse_args()

    # 1. Dependencies (Kaggle has torch; nibabel/scipy may need install).
    if not args.skip_deps:
        run([sys.executable, "-m", "pip", "install", "--quiet",
             "nibabel", "scipy"])

    # 2. Model code (clone if the folder is not already there).
    if not (args.model_dir / "train.py").exists():
        run(["git", "clone", "--depth", "1", GITHUB_REPO, "KidneyScan-AI"])
        args.model_dir = Path("KidneyScan-AI") / "model"
    print(f"model code: {args.model_dir.resolve()}")

    # 3. Data download (Hugging Face, Lite set). Skipped if present.
    # Lite layout is Images/<case>.nii.gz + Masks/<case>.nii.gz;
    # train.py reads that layout directly, no conversion step needed.
    if not args.skip_download and not list(args.data_dir.glob("case_*")) \
            and not list((args.data_dir / "Images").glob("*.nii.gz")):
        print(f"downloading {HF_DATASET} (Lite, ~40.9 GB) — this takes a while")
        run([sys.executable, "-m", "pip", "install", "--quiet",
             "huggingface_hub"])
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=HF_DATASET, repo_type="dataset",
                          local_dir=str(args.data_dir))
    n_cases = len(list(args.data_dir.glob("case_*"))) + len(
        list((args.data_dir / "Images").glob("*.nii.gz")))
    print(f"data: {args.data_dir.resolve()} ({n_cases} cases found)")

    # 4. Train (in the model folder, so `from light_unet3d import ...` works).
    cmd = [sys.executable, "train.py",
           "--data-dir", str(args.data_dir.resolve()),
           "--phase", args.phase,
           "--out-dir", str(Path("checkpoints").resolve())]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    run(cmd, cwd=str(args.model_dir))

    print("done. Download ./checkpoints/ from the Kaggle output tab "
          "before the session ends.")


if __name__ == "__main__":
    main()
