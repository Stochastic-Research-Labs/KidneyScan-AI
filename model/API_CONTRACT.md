# KidneyScan AI — API Contract

**This file is only a pointer. The canonical contract lives at
`../API_CONTRACT.md` — read and edit that file, not this one.**

The old model-side contract (`POST /predict`, field `file`, mask as a
`.nii.gz` file, `finding` string, scores including background) and the old
web-side contract (`POST /api/predict`, field `ct`, `mask_base64` preview,
`findings` object, scores without background) were merged into one
canonical contract v1.1 on 15 Nov 2026. See `../API_CONTRACT.md` for the
merged version both sides implement.
