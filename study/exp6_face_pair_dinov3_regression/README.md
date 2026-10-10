# Frozen DINOv3 Face-Pair Laboratory Delta Regression

Independent comparison of the current native224 Exp6 EfficientNet-B0 main
regression with frozen official DINOv3 ViT-S/16 plus head32 or head64. It does
not rebuild the main dataset, search a new split, or overwrite baseline results.

## Architecture

```text
Early RGB face (3,224,224) -> frozen shared DINOv3-S -> CLS (384)
Late RGB face  (3,224,224) -> same frozen encoder  -> CLS (384)
Late CLS - early CLS
  -> FC(384,32 or 64) -> LayerNorm -> SiLU -> Dropout(0.25) -> FC(32 or 64,1)
  -> robust-scaled laboratory delta (late value minus early value)
```

Backbone: 21,601,152 frozen parameters. Heads: 12,417 and 24,833 parameters.
Every target and hidden width has its own independently optimized head; there
is no shared trainable multitask head and no additional encoder adaptation.
`models.FrozenDeltaPredictor` is the equivalent online model. Checkpoints save
head parameters plus the common encoder SHA256/revision, not duplicate encoders.

## Controlled Clinical Protocol

- Exact main `outputs/face224` task CSVs, train-only raw-delta median/IQR, patient
  assignments and seed (20260808) are copied/verified without another search.
- Eleven indicators: O2Hb fraction, lactate, urea, troponin, platelets,
  hemoglobin, A/a PO2 ratio, creatinine, total bilirubin, HCT and reported eGFR.
- Both timepoints use main same-admission nearest-assay matching within 24h,
  validated session time, consecutive unique laboratory events and increasing
  face-video times. The original twenty selected nonadjacent frames are paired
  by rank. These are separate capture windows, not simultaneous videos.
- All five views remain: original, hflip, 90% center crop, brightness +6%, and
  contrast +8%. Each frame pair uses the same view at both timepoints.
- One complete batch: 12 distinct laboratory **pairs** x 20 frame pairs x one
  view = 240 frame pairs (480 source faces). Every view appears once per epoch.
- Main inverse-patient-pair-count weights are computed per split identically.
  Training loss is weighted frame-pair SmoothL1 (beta=0.5), not pooled-pair loss.
  Two microbatches of 120 frame pairs share one weighted objective and optimizer
  update; no final partial batch is dropped.
- Evaluation averages the twenty original-view predictions per pair, then
  inverse-scales. The actual raw delta is ground truth (not inverse float32
  labels). Comparison audits require identical test pair IDs, patients and values.

## Optimization And Comparability

Both DINO widths use the same settings as the existing frozen-DINO controls:
one head-only stage, AdamW lr=2e-4, weight decay=1e-3, cosine floor=1e-6,
maximum 80 epochs, validation-MAE patience=12, gradient clip=1, float16 AMP.
No warmup, backbone fine-tuning or torch compilation. The main comparator
instead trains EfficientNet in two stages; this is a representation/training
strategy comparison, **not** a backbone-only controlled ablation. Clinical
data, loss weights, batches and evaluation are held fixed. Head32 versus
head64 changes only hidden width.

## Storage And Launch

Frozen CLS features are label-independent. Exact matching video paths,
codecs, packet offsets, source-frame indices and encoder/view hashes permit
reuse of existing Exp2 DINO features; only additional videos are encoded.
The compact cache is about 215 MB for 1,401 videos x20 frames x5 views x384
float32 features. It is shared by all 22 heads; no pixel or video copies are
created. A small GPU smoke test checks both widths and exact view preprocessing.

```bash
python -m unittest study.exp6_face_pair_dinov3_regression.test_experiment
python -m study.exp6_face_pair_dinov3_regression.run --check-only
bash study/exp6_face_pair_dinov3_regression/launch_screen.sh
screen -r exp6_dinov3_regression
```

Four GPU workers dynamically consume the 22-job queue. Valid completed jobs
are reused on relaunch, incomplete jobs restart; incompatible source contracts
fail rather than overwrite results. The current other experiments are not stopped.

- Root log: `logs/run.log`; per-head logs: `outputs/head{32,64}/runs/<target>/train.log`.
- Outputs: `outputs/head32/`, `outputs/head64/` (metrics, predictions, histories,
  validation-selected checkpoints and per-width figures).
- Comparison PNG/PDF: `outputs/figures/`; values: `outputs/test_comparison.csv`.
- Audit CSV: `outputs/paired_test_audit.csv`; provenance: `outputs/experiment_manifest.json`.
- Automatic plots: loss/MAE histories with independently scaled correlation
  axes; observed-vs-predicted delta scatter with identity and linear fit;
  MAE, RMSE, r, R2, explained variance and direction bACC/AUROC comparison bars.
- `outputs/COMPLETE` is written only after all jobs and comparison figures succeed.

## Frozen EfficientNet Head32/Head64 Controls

The companion EN-B0 controls also freeze all encoder parameters and BatchNorm
statistics. Each face produces a 1280-dimensional GAP feature; late minus early
features enter a 32/64-hidden head with 41,089/82,177 trainable parameters.
The DINO single-stage optimizer, lr, scheduler, 80-epoch limit, patience=12,
dropout, weighted frame-pair loss and 120-pair microbatch accumulation are
reused unchanged. There is no second stage or feature standardization.

```bash
bash study/common/launch_frozen_en_regression_controls_screen.sh
screen -r frozen_en_exp2_exp6_regression
```

The combined four-GPU queue trains 20 new Exp2 and 22 new Exp6 frozen EN heads.
Exp6 outputs: `outputs/frozen_en_b0/head32/` and `outputs/frozen_en_b0/head64/`.
The shared cached features are under `study/common/cache/efficientnet_b0_frozen_exp2_exp6/`;
only one 717 MB numeric cache is created, with no copied images or videos.
After training, `outputs/figures/five_model_*` compares fine-tuned EN-B0,
frozen EN-B0 head32/head64, and frozen DINOv3 head32/head64 on identical test
pairs. Existing three-model figures and all existing checkpoints are retained.
Comparison values/audit: `outputs/five_model_test_comparison.csv` and
`outputs/five_model_test_audit.csv`. Root log:
`study/common/logs/frozen_en_regression_controls/run.log`.
