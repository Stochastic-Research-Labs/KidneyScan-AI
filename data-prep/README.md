# KidneyScan AI — Data Prep

Data: KiTS23 kidney tumor + cyst, 3D CT.
489 training cases. 110 test cases (masks private).
Labels: 0 background, 1 kidney, 2 tumor, 3 cyst.

WARNING: assistive tool only.
A doctor must confirm. Not a final diagnosis.
Not for real medical use.

## Files

- `download.py` — download KiTS23.
  Default: 5 Lite cases only.
- `check_data.py` — check .nii.gz files.
  Shape 512x512, labels, counts, missing files.
  Makes JSON + text report.
- `preprocess.py` — clip, normalize, crop to kidney, resample.
  Makes nnU-Net layout: imagesTr, labelsTr, dataset.json.
- `splits.py` — split 400 train / 89 val.
  Stratified by tumor size (<4 cm vs >=4 cm) and cyst yes/no.
  Seed 42. Also has Dice-by-size helpers.
- `test_synthetic.py` — test all logic on a tiny fake volume.
  No real data needed.

## How to run (on Kaggle)

```bash
pip install nibabel numpy scipy huggingface_hub
python3 download.py --lite --n-cases 5 --out ./raw
python3 check_data.py --data ./raw --out ./data_check.json
python3 preprocess.py --data ./raw --out ./preprocessed
python3 splits.py --stats ./data_check.json --out ./splits
python3 test_synthetic.py
```

Full data (only when ready):
```bash
python3 download.py --lite --n-cases 0 --out ./raw --yes   # all 489, ~40.9 GB
```

## Reuse table

We reuse ideas and code from others.
We do NOT copy-paste and call it ours.
Novelty stays ours: light model + cyst/small-tumor focus
+ edge loss + robustness test, checked by ablation.

| Source | License | What we reused | What we changed | How to cite |
|---|---|---|---|---|
| nnU-Net v2, https://github.com/MIC-DKFZ/nnUNet | Apache-2.0 | CT clip to 0.5/99.5 percentiles + z-score; resample to one spacing; crop to region; imagesTr/labelsTr/dataset.json layout | We wrote our own small functions, no nnU-Net code copied. Percentiles per case, not whole dataset. Crop uses kidney mask + margin. | Isensee et al., Nature Methods 2021, DOI 10.1038/s41592-020-01008-z |
| MONAI Auto3DSeg / SegResNet, https://github.com/Project-MONAI/MONAI | Apache-2.0 | Recipe idea only, for later training baseline. No code used in this folder yet. | Nothing yet. If used later, record it here. | Myronenko et al., arXiv 2310.04110 (KiTS23 winner) |
| KiTS23 official, https://github.com/neheller/kits23 | Code MIT, data CC BY-NC-SA 4.0 | Official download route (`kits23_download_data`), official metrics idea, dataset v0.1.4 | We added a 5-case test mode and a size guard. Full download is 94.8 GB. | Heller et al., KiTS21, arXiv 2307.01984; https://kits-challenge.org/kits23/ |
| KiTS23-Lite, https://huggingface.co/datasets/YongchengYAO/KiTS23-Lite | CC BY-NC-SA 4.0 | 489 images + masks, same as official v0.1.3, files renamed by case ID | None. We only download it. | Same KiTS23 cite as above + link to the Lite page |
| NeuroScan (own project) | Own work | 3D U-Net + FastAPI + Vercel demo pattern, for reference only | New organ (kidney), new labels (cyst added), light model | Not a public cite. Internal reference. |

Task048 v1 weights and nnssl checkpoints:
for testing / finetuning only, if needed later.
Task048 has kidney + tumor only, no cyst.
It is KiTS19, not KiTS23. Not used in this folder.

## What was NOT downloaded or run, and why

- No real KiTS23 data downloaded here.
  This run only wrote code and tested it on fake data.
  Full Lite is ~40.9 GB. Official full is 94.8 GB.
  Download 5 cases first on Kaggle, check them, then get more.
- No training run here. PyTorch is not installed on this machine.
  Training runs on Kaggle GPU later, in the model step.
- No Dice scores yet. No results are claimed here.
  Real Dice comes only from real data + real training.
- No push, no deploy, no post. Code only, in this folder.

## Gaps this data step serves (all six)

1. Cyst as own class (label 3), counted apart.
2. Small tumors <4 cm vs >=4 cm, split + scored apart.
3. Light model: small cropped volumes, so 1 GPU is enough.
4. Sharp edges: Surface Dice + HD95 scored later, on this split.
5. Robustness: same split used for noise / low-res / missing-slice tests.
6. Free 3D demo: preprocessed data feeds the demo later.
