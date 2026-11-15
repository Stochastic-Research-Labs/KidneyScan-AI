#!/usr/bin/env python3
"""Download KiTS23 data for KidneyScan AI.

Sources (see README.md reuse table):
  * Official: https://github.com/neheller/kits23 (code MIT, data CC BY-NC-SA 4.0)
    Dataset v0.1.4. Its installer downloads imaging from the Hugging Face
    dataset ``neheller/KiTS-Challenge-Imaging`` into a ``dataset/`` folder.
  * Lite mirror: https://huggingface.co/datasets/YongchengYAO/KiTS23-Lite
    KiTS23 v0.1.3, 489 images + masks, no change to any image or mask,
    files renamed by case ID. License CC BY-NC-SA 4.0. ~40.9 GB for all 489.

This script downloads NOTHING by itself in a normal run unless you ask for
cases. Default is 5 test cases only. Full download is 94.8 GB (official)
or ~40.9 GB (Lite). A guard stops any run over 20 cases unless --yes
is given, so a typo cannot start a 40 GB download.

Usage:
    python3 download.py --lite --n-cases 5 --out ./raw
    python3 download.py --lite --n-cases 0 --out ./raw --yes   # all 489 Lite cases
    python3 download.py --full --n-cases 5 --out ./raw         # official route

Needs: ``pip install huggingface_hub`` for the Lite route.
For the official route the ``kits23`` package must be installed first:
    git clone https://github.com/neheller/kits23
    cd kits23 && pip install -e .
then this script calls its ``kits23_download_data`` entry point.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

LITE_REPO = "YongchengYAO/KiTS23-Lite"
OFFICIAL_REPO = "neheller/KiTS-Challenge-Imaging"
OFFICIAL_GITHUB = "https://github.com/neheller/kits23"
TOTAL_CASES = 489
GUARD_LIMIT = 20


def _manifest(out_dir, source, repo, case_ids, n_requested):
    data = {
        "project": "KidneyScan AI",
        "dataset": "KiTS23",
        "source": source,
        "repo": repo,
        "license": "CC BY-NC-SA 4.0",
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "n_requested": n_requested,
        "n_downloaded": len(case_ids),
        "case_ids": case_ids,
        "cite": "Heller et al., KiTS23, https://kits-challenge.org/kits23/",
    }
    path = os.path.join(out_dir, "download_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def _pair_stem(filename):
    """Return the case stem for an image/mask file, or None."""
    name = os.path.basename(filename)
    if not name.endswith(".nii.gz"):
        return None
    stem = name[: -len(".nii.gz")]
    # Lite / nnU-Net style names end in _0000 (image) or have no suffix (mask).
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return stem


def download_lite(out_dir, n_cases):
    """Download the first ``n_cases`` cases from KiTS23-Lite via HF Hub.

    File names on the hub are discovered at run time (list_repo_files),
    not hard-coded, because the mirror renames files by case ID.
    Images and masks are paired by their shared stem.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.", file=sys.stderr)
        print("Run: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi()
    print(f"Listing files in {LITE_REPO} ...")
    files = api.list_repo_files(repo_id=LITE_REPO, repo_type="dataset")
    nii = [f for f in files if f.endswith(".nii.gz")]
    if not nii:
        print("ERROR: no .nii.gz files found on the hub repo.", file=sys.stderr)
        return 1

    # Split into images and masks by folder name when present, else by stem.
    images = [f for f in nii if "/Images" in f or f.lower().startswith("images")]
    masks = [f for f in nii if "/Masks" in f or f.lower().startswith("masks")]
    if not images or not masks:
        # Flat layout: every file is either an image or a mask; pair by stem
        # is impossible without names, so stop instead of guessing.
        print("ERROR: could not tell Images apart from Masks on the hub.",
              file=sys.stderr)
        print("Files seen (first 10):", nii[:10], file=sys.stderr)
        return 1

    img_by_stem = {_pair_stem(f): f for f in sorted(images)}
    msk_by_stem = {_pair_stem(f): f for f in sorted(masks)}
    stems = sorted(set(img_by_stem) & set(msk_by_stem))
    print(f"Found {len(stems)} cases with both image and mask.")
    if n_cases == 0:
        wanted = stems
    else:
        wanted = stems[:n_cases]
    if len(wanted) < (n_cases or len(stems)):
        print(f"WARNING: only {len(wanted)} cases available.", file=sys.stderr)

    os.makedirs(os.path.join(out_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "Masks"), exist_ok=True)
    done = []
    for stem in wanted:
        for src, sub in ((img_by_stem[stem], "Images"),
                         (msk_by_stem[stem], "Masks")):
            hf_hub_download(repo_id=LITE_REPO, filename=src,
                            repo_type="dataset", local_dir=out_dir)
        done.append(stem)
        print(f"  [{len(done)}/{len(wanted)}] {stem}")
    path = _manifest(out_dir, "lite", LITE_REPO, done, n_cases)
    print(f"Done. {len(done)} cases in {out_dir}. Manifest: {path}")
    return 0


def download_official(out_dir, n_cases):
    """Download via the official kits23 package (full 94.8 GB dataset).

    The official installer downloads ALL cases; it has no per-case option.
    So --n-cases is ignored here on purpose, and we refuse unless --yes
    was given (the caller already confirmed the full size).
    """
    print("Official route: full KiTS23 is 94.8 GB, all cases at once.")
    print("Source:", OFFICIAL_GITHUB)
    result = subprocess.run(["kits23_download_data"], cwd=out_dir)
    if result.returncode != 0:
        print("ERROR: kits23_download_data failed.", file=sys.stderr)
        print("Install the official package first:", file=sys.stderr)
        print("  git clone https://github.com/neheller/kits23", file=sys.stderr)
        print("  cd kits23 && pip install -e .", file=sys.stderr)
        return result.returncode or 1
    path = _manifest(out_dir, "official-full", OFFICIAL_REPO, [], n_cases)
    print(f"Done. Manifest: {path}")
    print("NOTE: official download has no per-case limit; verify with")
    print("  python3 check_data.py --data <out>/dataset")
    return 0


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Download KiTS23 data (test-size by default).")
    ap.add_argument("--lite", dest="lite", action="store_true", default=True,
                    help="Use KiTS23-Lite on Hugging Face (default).")
    ap.add_argument("--full", dest="full", action="store_true",
                    help="Use the official kits23 installer (full 94.8 GB).")
    ap.add_argument("--n-cases", type=int, default=5,
                    help="Cases to download. 0 = all 489 (Lite only). Default: 5.")
    ap.add_argument("--out", default=os.path.join(here, "raw"),
                    help="Output folder. Default: ./raw next to this script.")
    ap.add_argument("--yes", action="store_true",
                    help="Confirm a download over 20 cases / the full dataset.")
    args = ap.parse_args(argv)

    if args.n_cases < 0:
        print("ERROR: --n-cases must be 0 or more.", file=sys.stderr)
        return 2
    if args.full:
        if not args.yes:
            print("ERROR: --full downloads 94.8 GB. Re-run with --yes to confirm.",
                  file=sys.stderr)
            return 2
        os.makedirs(args.out, exist_ok=True)
        return download_official(args.out, args.n_cases)
    if args.n_cases > GUARD_LIMIT and not args.yes:
        print(f"ERROR: --n-cases {args.n_cases} is over the {GUARD_LIMIT}-case guard.",
              file=sys.stderr)
        print("Re-run with --yes if you really want that many cases.", file=sys.stderr)
        return 2
    if args.n_cases == 0:
        print("NOTE: --n-cases 0 downloads all 489 Lite cases (~40.9 GB).")
    os.makedirs(args.out, exist_ok=True)
    return download_lite(args.out, args.n_cases)


if __name__ == "__main__":
    sys.exit(main())
