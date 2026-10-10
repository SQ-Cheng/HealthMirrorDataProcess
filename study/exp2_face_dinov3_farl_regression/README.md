# Frozen DINOv3 + FaRL Laboratory Regression

Independent parallel experiment; no existing results are overwritten. Ten
current 24h main targets, exact patient splits, twenty native224 selected
frames, five views, train-only raw-label median/IQR, distinct-lab batches and
frame SmoothL1 are copied/verified against frozen DINO head32. One independent
model per analyte and variant: **20 regression models**.

## Encoders And Input

The same augmented RGB frame goes to two entirely frozen/eval encoders:

- Official DINOv3 ViT-S/16: final normalized CLS, 384 dimensions; existing
  verified cache is reused without extraction, training or target fitting.
- Local `FaRL-Base-Patch16-LAIONFace20M-ep64.pth`: official CLIP ViT-B/16 image
  architecture, 768-wide twelve-block visual trunk and pretrained projection
  to 512 dimensions. Every inference tensor is loaded strictly. Only the 17
  known MIM branch tensors (`mask_token`, `lm_transformer`, `lm_head`, `ln_lm`)
  and text tensors are intentionally unused, never random-filled.

FaRL is used as described by the official FaceCLIP instructions:
https://github.com/FacePerceiver/FaRL#use-farl-as-faceclip

Use the official OpenAI CLIP implementation pinned to commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. No baseline CLIP checkpoint download
is required. The visual state alone has 86,192,640 frozen parameters. DINO
uses existing ImageNet normalization; FaRL uses CLIP mean
(0.48145466,0.4578275,0.40821073) and std
(0.26862954,0.26130258,0.27577711). Native224 originals are not resized;
the same defined center-crop view is shared. No additional face alignment or
feature L2 normalization is introduced.

## Two Heads

```text
Concat:
  concat(DINO384, FaRL512) -> FC(896,32) -> LN -> SiLU -> Dropout(.25) -> FC(32,1)

Gated:
  DINO384 -> FC(384,64) -> LN -> SiLU -> zd
  FaRL512 -> FC(512,64) -> LN -> SiLU -> zf
  g = sigmoid(FC(concat(zd,zf),64))
  z = g*zd + (1-g)*zf
  z -> FC(64,32) -> LN -> SiLU -> Dropout(.25) -> FC(32,1)
```

Trainable parameter counts: concat **28,801**; gated **68,161** (including
both projectors and gate). The gate starts at 0.5 for every dimension. This
is a feature-fusion comparison, not a parameter-count-matched ablation.
Head hidden dimension stays 32; projected expert dimension is 64.

## Training And Artifacts

Same DINO single-stage schedule: AdamW lr=2e-4, weight decay=1e-3, cosine
floor=1e-6, max 80 epochs, val raw-unit MAE patience=12, gradient clip=1,
AMP, no compile/warmup/backbone unfreezing. A full batch is twelve different
clinical lab events x20 frames xone view =240 frame losses. All five views
are retained across an epoch. Evaluation averages twenty original-frame
predictions per video. FaRL features are extracted once (~285 MB) and
shared; no concatenated feature cube, pixel cache or per-task encoder copies.

```bash
pip install --no-deps -r study/exp2_face_dinov3_farl_regression/requirements.txt
python -m unittest study.exp2_face_dinov3_farl_regression.test_experiment
bash study/exp2_face_dinov3_farl_regression/launch_screen.sh
screen -r exp2_dinov3_farl_regression
```

Four GPU workers extract features, then dynamically consume head jobs. The
launcher runs a small real-feature optimizer/checkpoint smoke test before
formal training. Valid completed jobs resume; incompatible contracts fail.

- Log: `logs/run.log`; per-target logs in `outputs/regression/<variant>/runs/<target>/train.log`.
- Results: `outputs/regression/{concat,gated}/` with checkpoints, all histories,
  metrics, per-video predictions and per-variant PNG/PDF.
- Automatic paired baseline comparisons: `outputs/figures/`,
  `outputs/test_comparison.csv`, `outputs/paired_test_audit.csv`.
- The cached pixels/views/index and all held-out video IDs/labels are verified
  before comparing the existing DINO head32 with concat and gated models.
