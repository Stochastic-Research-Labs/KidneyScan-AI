# KidneyScan AI — Literature Review
Date: 15 Nov 2026
Project: Kidney tumor + cyst in 3D CT, KiTS23
Data: KiTS23, 599 cases, 489 train, 110 test
Labels: 0 background, 1 kidney, 2 tumor, 3 cyst
Goal: Q1 journal + helpful for others, not just learning
Rule: assistive only. A doctor must confirm. Not a final diagnosis.

## 1-page summary — what people did, what is missing, what we will make

What people did:
- KiTS19 (2019): 210 train, 90 test, 106 teams. Winner used nnU-Net.
- Kidney Dice 0.974, tumor Dice 0.851. Kidney is almost solved. Tumor is not.
- KiTS21 (2021): added cyst as a third label. Test came from a new hospital.
- KiTS23 (2023): 489 train, 110 test. Winner used MONAI Auto3DSeg, 15 models.
- Average Dice 0.835, surface Dice 0.723. That average hides a big problem: cyst is very low.
- Cyst Dice is only 0.18 to 0.29 with a normal U-Net. Tumor is 0.58 in one 2024 study.
- Big models win, but they need 8 big GPUs and 15 models. They are not light.
- Models drop a lot on new hospital data. One KiTS model fell from 0.85 to 0.66.

What is missing:
- Small tumors and cysts are still bad.
- Cyst is often mixed up with tumor.
- No light model that is good and fast on one Kaggle GPU.
- Sharp edges are not scored well. Most papers report Dice only.
- No simple test for bad input: noise, low resolution, missing slices.
- No free 3D demo for doctors and students.

What we will make:
- One light 3D U-Net, good at small tumors and cysts, with sharp edges.
- Test it by tumor size, by cyst, by edge score, and by bad input.
- Show it in a free 3D demo, like NeuroScan.
- Compare it fairly with nnU-Net on the same split. Show what each change adds.

## Can we finetune the ~0.97 winner?

Short answer: code yes, ready model no, and finetune alone is not enough for Q1.

KiTS19 winner — Isensee team, nnU-Net:
- Code: yes, public. https://github.com/MIC-DKFZ/nnUNet
- License: Apache 2.0. Stars: 9,043 on the page I opened.
- Paper: Isensee et al., Nature Methods, 2021, 18(2), 203-211.
- Dice: kidney 0.974, tumor 0.851.
- Weights, v1: yes, Task048 KiTS_clean, kidney + tumor, CT.
- Command in v1: `nnUNet_download_pretrained_model Task048_KiTS_clean`
- Weights, v2: no. nnU-Net v2 does not ship pretrained weights any more.
- V2 does offer finetune from nnssl self-supervised checkpoints.
- Limit: Task048 has kidney + tumor only. No cyst. It is KiTS19, not KiTS23.

KiTS23 winner — Myronenko team, MONAI Auto3DSeg:
- Code: yes, public. https://github.com/Project-MONAI/MONAI
- Paper: Myronenko et al., arXiv 2310.04110, 6 Oct 2023. Preprint, MICCAI 2023, 1st place.
- Dice: average Dice 0.835, surface Dice 0.723.
- Method: ensemble of 15 models, 10 SegResNet + 5 DiNTS.
- Trained on 8-GPU 48GB NVIDIA A40 machine. That is in the paper, not in Kaggle free GPU.
- Weights: no single winning checkpoint link was found in the paper.
- You can rerun their Auto3DSeg recipe, but you cannot just download their winning model and finetune it.

Other finetune option — TransferSAM:
- Code: yes, public. https://github.com/Noza23/TransferSAM
- It tries to finetune Meta SAM for kidney + tumor + cyst.
- Weights: a KiTSAM.pkl file on Google Drive, linked in its README.
- Limit: 0 stars, course project, no Dice score in the README I opened. Do not use it as the main model.

So: use nnU-Net as the baseline to beat. Do not just finetune it and call it a paper.

## Table — what people did before

| Paper / year | Data | Method | Dice kidney / tumor / cyst | Code? | Weights? | Did well | Missed |
|---|---|---|---|---|---|---|---|
| Heller et al., KiTS19, Med Image Anal, 2021, 67:101821, DOI 10.1016/j.media.2020.101821 | KiTS19, 210 train, 90 test, 106 teams | Challenge report, top teams all 3D U-Net / nnU-Net | Kidney 0.974 / tumor 0.851, composite 0.912. Human kidney 0.983, tumor 0.923. | Data starter code: https://github.com/neheller/kits19 | No single winner checkpoint in the paper | First big public kidney CT set, clear test, human score to beat | Tumor gap to human is still 0.072. No cyst. No light model. No demo. |
| Isensee et al., nnU-Net, Nature Methods, 2021, 18(2), 203-211, DOI 10.1038/s41592-020-01008-z | Many medical sets, incl. KiTS | Self-configuring 3D U-Net pipeline, auto preprocessing + training | No single KiTS Dice in this paper. It is the framework the KiTS19 winner used. | Yes: https://github.com/MIC-DKFZ/nnUNet, Apache 2.0, 9,043 stars | v1: Task048 KiTS_clean via Zenodo. v2: no pretrained weights shipped. | Strong baseline, easy to run, still hard to beat | Heavy, slow, no cyst in v1 weights, no edge score, no robustness test |
| Heller et al., KiTS21, arXiv 2307.01984, 5 Jul 2023 | KiTS21, 300 train cases, test from a new hospital, 3 labels | Challenge report, top team coarse-to-fine nnU-Net | No single Dice in the abstract I opened. Top teams got closer to human. Test from a new hospital was harder. | Data: https://github.com/neheller/kits21 (link in paper refs) | No winner checkpoint in the abstract | Added cyst, added outside-hospital test, better annotation | Cyst scores were low. No paper in this table gives a good cyst Dice for KiTS21 alone. |
| Myronenko et al., KiTS23 winner, arXiv 2310.04110, 6 Oct 2023, preprint | KiTS23, 489 train, 110 test hidden | MONAI Auto3DSeg, SegResNet + DiNTS, 5-fold, ensemble 15 models | Average Dice 0.835, surface Dice 0.723, 1st place. | Yes, MONAI: https://github.com/Project-MONAI/MONAI | No single winning checkpoint found in the paper | Best average on KiTS23, auto recipe, good for non-experts | 15 models, 8x A40 GPUs. Too big for Kaggle free GPU. No per-class cyst Dice in the abstract. No light version. |
| Uhm et al., KiTS23 2nd, arXiv 2312.05528, 9 Dec 2023, preprint | KiTS23, 489 train | 3D U-Net, training settings + post-processing | Average Dice 0.820, surface Dice 0.712, 2nd place. | No code link in the abstract I opened | No | Shows simple 3D U-Net + good post-processing is close to the winner | Still a big 3D U-Net. No light model. No cyst-only score in the abstract. |
| Raman et al., external test, Eur J Radiol Artif Intell, 2025, 3:100035, DOI 10.1016/j.ejrai.2025.100035 | KiTS19 210 train, plus hospital data 555 train, 139 test | nnU-Net v2 3D, train on one site, test on the other | KiTS model: tumor Dice 0.85 on KiTS test, 0.66 on hospital test. Hospital model: 0.74 on its own test, 0.60 on KiTS test. | No code link in the abstract I opened | No | Proves the drop is real, with a big outside set | No fix for the drop. No cyst. No light model. |
| Scalco et al., uncertainty, Med Phys, 2024, DOI 10.1002/mp.17189 | KiTS21, 300 CT, kidney + tumor + cyst | Bayesian + ensemble + test-time augmentation, 4 methods | Kidney 0.92, tumor 0.58, cyst 0.21, average across methods. | No code link in the search result I opened | No | Shows cyst is the hardest class, and shows uncertainty matters | Cyst 0.21 is very low. No new model to fix it. No demo. |
| Saluja et al., BackSplit, arXiv 2511.19394, 24 Nov 2025, CVPR 2026 | KiTS23 cyst, N=489, 5-fold, plus 4 other sets | Split background into kidney + tumor + other, train cyst with that context. No extra params at test. | Cyst, U-Net: 0.1787 to 0.4573 with BackSplit. ResEnc: 0.2532 to 0.4614. SegResNet: 0.2914 to 0.4290. | No GitHub link in the arXiv abstract I opened. Check the full paper for code before you use it. | No | Best cyst evidence I found. Simple idea, big gain, no extra test cost | Still only 0.46 for cyst. It is a 2025 preprint / CVPR 2026 paper, not an old settled method. Needs your own test on KiTS23. |

Notes on the table:
- Dice is overlap, 0 to 1. 1 is perfect. 0 is no overlap.
- Surface Dice is edge overlap. It punishes bad borders more than Dice does.
- Average Dice in KiTS23 hides the cyst score. Always report kidney, tumor, and cyst apart.
- A training Dice is not a test Dice. WP-UNet, a ResearchSquare preprint, reports training Dice 0.98 and only 1,297,441 params. That is training only, not a test score. Do not quote it as a result.

## Research gaps — 4 to 6, each with evidence

Gap 1 — Cyst is very bad, and tumor is often mixed with cyst.
- Evidence: Scalco 2024, cyst Dice 0.21, tumor 0.58, kidney 0.92.
- Evidence: BackSplit, normal U-Net cyst Dice 0.1787 on KiTS23.
- Why it matters: cyst and tumor need different treatment. A wrong label is a wrong plan.

Gap 2 — Small tumors and small cysts fail first.
- Evidence: BackSplit abstract: small lesions are small, rare, and cause many false alarms.
- Evidence: KiTS19, tumor Dice 0.851, human tumor Dice 0.923. The gap is in tumor, not kidney.
- Why it matters: small tumors are the ones doctors want to catch early.

Gap 3 — No light model that is good on one free GPU.
- Evidence: KiTS23 winner used 15 models and 8x A40 48GB GPUs. That is in the Myronenko paper methods.
- Evidence: nnU-Net is strong, but it is a full pipeline, not a small model for a demo.
- Why it matters: small hospitals and students do not have 8 big GPUs. Kaggle gives 30 hours a week.

Gap 4 — Sharp edges are not tested well.
- Evidence: KiTS23 winner reports surface Dice 0.723, lower than Dice 0.835.
- Evidence: KiTS23 2nd place, surface Dice 0.712, Dice 0.820.
- Why it matters: a tumor with a bad edge gives a wrong volume. Volume is used for surgery planning.

Gap 5 — Models drop on new hospital data.
- Evidence: Raman 2025, KiTS model tumor Dice 0.85 on KiTS, 0.66 on a new hospital set.
- Why it matters: a model that only works on one hospital is not helpful for others.

Gap 6 — No free 3D demo for doctors and students.
- Evidence: all papers above give Dice and code, but none give a live upload + 3D viewer for a doctor to try.
- Why it matters: Wajid already has this for NeuroScan. No KiTS paper in this review has it. This is a use gap, not a Dice gap.

## Recommended gap — ONE gap for Wajid

Pick this one:
Light 3D U-Net, good at small tumors and cysts, with sharp edges, and good on bad input.

Full name for the paper:
Lightweight multi-class 3D segmentation of kidney, tumor, and cyst on KiTS23, with edge-aware training and robustness testing.

Why this one:
- It merges Wajid's own gaps: light model + sharp edges + robustness on bad input.
- It uses his NeuroScan skill: 3D U-Net, same pipeline, new organ.
- It fits Kaggle: one light model, 12 to 20 GPU hours, not 15 models on 8 GPUs.
- It is testable in 3 to 4 weeks, part time.
- It helps others: small hospitals need a fast, light model, not a big ensemble.
- It is new: no paper in the table does light + cyst + edge + robustness together on KiTS23.

What is new vs just finetuning the winner:
- Finetune alone: same model, same data, a little better Dice. Reviewers will say no novelty.
- This plan: 3 small changes, each tested alone.
- Change 1: multi-class context. Train kidney + tumor + cyst together. Cyst is not background. This is the BackSplit idea, in a simple form.
- Change 2: light model. Fewer channels, depthwise blocks, about 1 to 5 million params. Measure params, speed, and GPU memory. Compare with nnU-Net baseline.
- Change 3: edge-aware loss. Dice + cross-entropy + boundary loss. Score with Surface Dice and HD95, not Dice only.
- Change 4: robustness test. Add noise, lower resolution, drop 10% of slices. Report how much Dice falls. A good model falls less.

Experiments to run, in order:
1. Baseline: light 3D U-Net, KiTS23, 400 train, 89 test. Report Dice per class, Surface Dice, HD95, params, time.
2. Baseline + multi-class context. Same split, same metrics. Did cyst go up?
3. Baseline + edge loss. Same split. Did Surface Dice and HD95 go up?
4. Full model: context + light + edge loss. Same split.
5. Ablation table: remove one change at a time. Show what each change adds.
6. Size split: tumor <4 cm vs >=4 cm. Cyst present vs no cyst. Report Dice apart.
7. Robustness split: clean vs noisy vs low resolution vs missing slices. Report Dice drop.
8. Comparison: same split, nnU-Net baseline. Do not compare your test Dice with their hidden test Dice. Compare on the same 89 cases.

What we will make:
- Light model code + Kaggle notebook, public on GitHub.
- Model file (.pt), with training curves and a Dice report.
- Live demo: upload CT, model runs, 3D kidney grey, tumor yellow/red, cyst blue, rotate 360, volume in cm3.
- Paper draft: problem, gap, method, results, ablation, limits, what failed.
- Limits to write honestly: cyst is still hard, one dataset only, no real hospital test yet, assistive only.

Timeline, 3 to 4 weeks, part time:
- Week 1: data check, 20-case test, baseline runs.
- Week 2: full train, 400 train, 89 test, ablation part 1.
- Week 3: edge + robustness tests, demo build.
- Week 4: paper draft, demo video, LinkedIn draft for Wajid to check.

## How Wajid can defend a Q1 paper with zero knowledge now

Honest answer: you do not need to know everything now. You need to know your own work very well.
- Read only 3 papers first, in this order: Heller KiTS19, Isensee nnU-Net, Myronenko KiTS23 winner.
- For each paper, write 1 page: problem, data, method, result, gap. In simple English.
- Learn by doing: run the baseline first. You will learn the terms when you see the code fail and fix it.
- Keep a glossary: Dice, Surface Dice, HD95, ablation, cross-validation, overfit. One line each.
- For defense, know these 6 questions cold: why KiTS23, why light, why multi-class, why edge loss, why your split, what failed.
- Bring your supervisor in early: Mam Saira Ishtiaq, and Hamza Mustafa. A Q1 paper needs a guide. Do not submit alone.
- Q1 rule: finetune alone is not enough. You need novelty + fair comparison + ablation + honest limits + a test on new data.

## Sources — links I opened and checked

- KiTS19 paper, arXiv: https://arxiv.org/abs/1912.01054
- KiTS19 paper, PubMed: https://pubmed.ncbi.nlm.nih.gov/33049579/
- nnU-Net code: https://github.com/MIC-DKFZ/nnUNet
- nnU-Net paper DOI: 10.1038/s41592-020-01008-z
- KiTS21 paper, arXiv: https://arxiv.org/abs/2307.01984
- KiTS23 data code: https://github.com/neheller/kits23
- KiTS23 official site: https://kits-challenge.org/kits23/
- KiTS23 winner, arXiv: https://arxiv.org/abs/2310.04110
- KiTS23 2nd, arXiv: https://arxiv.org/abs/2312.05528
- MONAI code: https://github.com/Project-MONAI/MONAI
- BackSplit, arXiv: https://arxiv.org/abs/2511.19394
- External test, DOI: 10.1016/j.ejrai.2025.100035
- Uncertainty, DOI: 10.1002/mp.17189
- TransferSAM code: https://github.com/Noza23/TransferSAM

Status of each paper:
- Published, peer-reviewed: Heller KiTS19 (Med Image Anal 2021), Isensee nnU-Net (Nature Methods 2021), Raman external test (2025), Scalco uncertainty (Med Phys 2024).
- Preprint, not yet a journal paper: Myronenko KiTS23 winner (arXiv 2023), Uhm KiTS23 2nd (arXiv 2023), Heller KiTS21 (arXiv 2023), Saluja BackSplit (arXiv 2025, CVPR 2026).
- Do not cite a preprint as settled fact. Cite it as a preprint with its arXiv number.
