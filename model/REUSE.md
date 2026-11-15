# KidneyScan AI — Reuse Table (REUSE.md)

Date: 15 Nov 2026. Rule (user, 23:10 PKT): reuse efficiently, do not
rewrite from scratch. Record source + license + what was reused +
what changed + how to cite. Novelty must stay clear.

Our novelty (not copied from anyone): ONE light 3D U-Net (~1-5M params)
that does kidney + tumor + cyst together, with a boundary/edge loss,
cyst-aware class weighting, and ablation + robustness testing on the
same 400/89 KiTS23 split. No single source below does light + cyst +
edge + robustness together on KiTS23.

## 1. nnU-Net v2 — preprocessing + training pipeline pattern, baseline

- Source: https://github.com/MIC-DKFZ/nnUNet
- License: Apache 2.0 (free to use, keep the license notice if code is copied).
- Paper to cite: Isensee et al., Nature Methods, 2021, 18(2), 203-211.
  DOI: 10.1038/s41592-020-01008-z
- What we reused: the pipeline IDEAS — HU clipping + normalisation,
  crop to foreground region, patch-based training, Dice+CE loss,
  mixed precision, checkpoint/resume, fixed train/val split.
  `train.py` follows this pattern; no nnU-Net code was copied into this
  folder. Install it separately for the baseline run:
  `pip install nnunetv2` (see ablation run F).
- What we changed and why: nnU-Net is a heavy full pipeline (its KiTS
  configs use 30M+ params and long training). We use a light 3D U-Net
  (~1.68M params at base 48, measured) so it fits 1 Kaggle GPU, add a cyst class
  (nnU-Net v1 Task048 weights have kidney+tumor only, no cyst), and add
  an edge loss + Surface Dice/HD95 scoring, which nnU-Net does not
  report by default.
- Weights: nnU-Net v2 ships NO pretrained weights. v1 Task048_KiTS_clean
  (kidney+tumor only, no cyst) may be used for TESTING ONLY via
  `nnUNet_download_pretrained_model Task048_KiTS_clean` — it cannot be
  the cyst model. v2 offers finetuning from nnssl self-supervised
  checkpoints instead; that is a follow-up, not used yet.

## 2. MONAI — Auto3DSeg / SegResNet recipes (reference only)

- Source: https://github.com/Project-MONAI/MONAI
- License: Apache 2.0.
- Paper to cite: Myronenko et al., arXiv 2310.04110 (KiTS23 winner,
  preprint, MICCAI 2023, 1st place). Also MONAI itself:
  Cardoso et al., arXiv 2202.02692.
- What we reused: the KiTS23 winner's recipe as a REFERENCE for what a
  strong model looks like (SegResNet + DiNTS ensemble, 15 models, 8x A40
  GPUs, average Dice 0.835, surface Dice 0.723). No MONAI code was
  copied into this folder.
- What we changed and why: the winner needs 15 models and 8 big GPUs —
  impossible on Kaggle free GPU. We build ONE light model instead and
  compare against the heavy approach honestly (ablation run F uses
  nnU-Net as the heavy baseline; a MONAI Auto3DSeg run is a follow-up
  if time allows).

## 3. KiTS23 data + starter code

- Source: https://github.com/neheller/kits23 and
  https://kits-challenge.org/kits23/
- License: CC BY-NC-SA 4.0 — free for research, NOT commercial.
- Paper to cite: Heller et al., KiTS19, Med Image Anal, 2021, 67:101821
  (challenge report); KiTS23 is the 2023 edition, 489 train / 110 test.
- What we reused: the dataset itself (CT + masks, labels 0 background,
  1 kidney, 2 tumor, 3 cyst) and the case layout
  (`case_XXXXX/imaging.nii.gz` + `segmentation.nii.gz`).
- What we changed: nothing in the data. Our split (400 train / 89 val,
  seed 42) is our own, saved in `checkpoints/split.json` by train.py.
- Hugging Face mirror for download: `neheller/KiTS-Challenge-Imaging`
  (full 94.8 GB; Lite 40.9 GB) — see the data worker's notes.

## 4. 3D U-Net architecture pattern

- Source pattern: Cicek et al., 3D U-Net, MICCAI 2016, arXiv 1606.06650.
- What we reused: the encoder-decoder + skip-connection SHAPE.
- What we changed: depthwise-separable 3D convolutions (MobileNet idea,
  Howard et al. 2017, arXiv 1704.04861) instead of standard 3x3x3 convs,
  GroupNorm instead of BatchNorm (small batches), base 16 channels.
  `light_unet3d.py` is written for KidneyScan from these patterns — it
  is not copied code from any repo.

## 5. BackSplit — cyst-context idea (idea only, no code)

- Source: Saluja et al., arXiv 2511.19394 (24 Nov 2025, CVPR 2026).
- What we reused: the IDEA — cyst is hard because it is small and rare;
  training it with kidney+tumor context (not as background) lifts cyst
  Dice from 0.18-0.29 to ~0.43-0.46. Our simple form: cyst as its own
  class + higher class weight + cyst-case oversampling (ablation run D).
- What we changed: BackSplit splits background into kidney/tumor/other
  as extra input channels. We do not do that yet — we test the simpler
  weighting/oversampling first. If D fails, BackSplit's full method is
  the follow-up. Cite it as a preprint (arXiv number), not settled fact.

## 6. Boundary loss — edge-aware training (idea + simple implementation)

- Source idea: Kervadec et al., Boundary loss, MICCAI 2019,
  arXiv 1812.07032 (distance-transform boundary loss).
- What we reused: the IDEA that Dice alone ignores edge errors.
- What we changed: our `losses.py` uses a fast gradient-magnitude edge
  loss (no scipy distance transform per batch) as a first version.
  The full Kervadec distance-transform loss is a follow-up if Surface
  Dice does not improve. Cite Kervadec et al. when the edge loss is used.

## 7. NeuroScan AI — the user's own 3D U-Net pipeline (reference)

- Source: the user's own NeuroScan project (BraTS 2021, 3D U-Net,
  40 epochs, PyTorch, Modal backend, Vercel demo).
- What we reused: the pipeline SHAPE — preprocess, train, inference
  returns masks + volumes, web demo shows original vs predicted.
- What we changed: new organ (kidney vs brain), new data (KiTS23 CT
  vs BraTS MRI), new class (cyst), light model, edge loss, ablation.
  No NeuroScan code was copied; the pattern is the user's own work.

## 8. TransferSAM — NOT used

- Source: https://github.com/Noza23/TransferSAM (finetune Meta SAM
  for kidney+tumor+cyst). 0 stars, course project, no Dice in README.
- Decision: NOT used as the main model. Listed here so nobody
  re-discovers it and assumes it is a ready model. It is not.

## How to cite (paper draft)

- Data: Heller et al. KiTS19 (Med Image Anal 2021) + KiTS23 site.
- Baseline/framework: Isensee et al. nnU-Net (Nature Methods 2021).
- Winner reference: Myronenko et al. arXiv 2310.04110.
- Cyst idea: Saluja et al. arXiv 2511.19394.
- Edge loss idea: Kervadec et al. arXiv 1812.07032.
- Architecture: Cicek et al. 3D U-Net (2016); Howard et al. MobileNet (2017).
- Licenses to respect: nnU-Net + MONAI Apache 2.0; KiTS23 CC BY-NC-SA 4.0
  (research only, no commercial use).
