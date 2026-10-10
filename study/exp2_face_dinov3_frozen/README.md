# Frozen DINOv3-S Face Regression And Classification

Sixteen independent heads predict the same eight current Exp2 targets: O2Hb
fraction, lactate, urea, total bilirubin, platelets, hemoglobin, A/a PO2 ratio,
and creatinine. "12 values" means twelve different matched laboratory events
per full batch, not twelve target indicators.

## Architecture And Freeze Contract

The official Meta DINOv3 ViT-S/16 LVD-1689M backbone has 21,601,152 frozen
parameters, patch size 16, embedding width 384, twelve blocks, six attention
heads, and four storage tokens. It is not ViT-S+ or a DINOv2 substitute.
The pinned upstream source revision is in `config.py`; the source cache lives
under `../common/backbones/dinov3/` and is ignored by Git.

Each native224 RGB frame produces its final normalized 384-dimensional CLS
vector. The scalar head is Linear(384,32) -> LayerNorm -> SiLU -> Dropout(0.25)
-> Linear(32,1): 12,417 trainable parameters. Each target/family has a separate
head. The encoder stays in eval mode with requires_grad=False and inference
without gradients. There is no backbone fine-tuning or second training stage.

Fixed original/hflip/center-crop/brightness/contrast views and ImageNet RGB
normalization exactly reuse the native224 pipeline. Frozen CLS features are
extracted once, shared by all heads, and saved as float32: about 163 MiB for
the current 22,260 source frames x five views. No patch cubes, decoded images,
new videos, or sixteen duplicate backbone checkpoints are stored. The cache
records source/weights/augmentation hashes. Head checkpoints record the shared
weight SHA256 and source revision, so predictions can be reproduced online.

## Data And Loss

The corrected 12h single patient-disjoint split and train-only median/IQR
regression scalers are reused without a new split search. Each full batch is
12 different lab events x 20 frames x 1 view = 240 images and twelve losses.
All five views remain present across different batches; a final partial batch
is retained. One event is identified by hospital ID plus the target's matched
lab-report timestamp. This does not require twelve different patients.

Classification uses weighted BCE on the mean of twenty frame probabilities,
with train-only negative/positive video-count pos_weight. Hb thresholds remain
male <130 g/L and female <120 g/L. Regression uses SmoothL1(beta=0.5) on mean
robust-scaled predictions; reported values are inverse-transformed to raw units.
The loss unit is video x view. Evaluation uses twenty original-view frames only.

## Single-Stage Head Training

AdamW, lr=2e-4, weight decay=1e-3, cosine decay to 1e-6, maximum 80 epochs,
patience=12, gradient clip=1, dropout=0.25. No warmup or compilation is used.
The frozen final encoder LayerNorm is retained; no target-specific feature
standardization is fitted. Early stopping selects validation bACC for
classification and raw-unit validation MAE for regression. Only heads enter
the optimizer. Four GPUs extract features and dynamically schedule head jobs.

All histories, train/val/test metrics, checkpoints, video predictions, regression
scatter plots, classification confusion matrices, and paired EfficientNet
comparisons are automatic. Eight-panel figures use four columns and two rows.
The matched EfficientNet control is the twelve-distinct-lab view-loss experiment.
This compares frozen DINOv3 representation/head fitting against fine-tuned EN-B0,
not an isolated freeze ablation of the same backbone.

## Required Authorized Weights

The ViT-S/16 checkpoint is now installed in the common weights directory and
has passed the full SHA256 check, strict loading, and a 224x224 CPU forward
test. SHA256: `08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d`.
The launch script selects the refreshed `lab_update_20261007` reference and
the overwritten paired EfficientNet controls; it does not use the earlier
12h cohort. No completed DINO results existed before this update.

Official sources:
- https://github.com/facebookresearch/dinov3
- https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m

At creation time, the official Hugging Face model returned GatedRepoError/401
and no local authorized checkpoint was available. Formal training never falls
back to randomized weights or an unofficial mirror. Obtain the official Meta
PyTorch checkpoint through the publisher's authorization process and place it at:

`../common/pretrained_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`

Alternatively, configure an authorized official Meta signed URL in the local
`DINOV3_WEIGHTS_URL` environment variable and run the helper below. Do not put
signed URLs or tokens in Git or logs. The checkpoint must pass the official
08c60483 SHA256 prefix and strict state-dict checks. HF safetensors cannot simply
be renamed to this file; this implementation expects the official PyTorch format.

```bash
python -m study.exp2_face_dinov3_frozen.backbone
python -m study.exp2_face_dinov3_frozen.run --check-only
python -m unittest study.exp2_face_dinov3_frozen.test_experiment
bash study/exp2_face_dinov3_frozen/launch_screen.sh
screen -r exp2_dinov3_frozen
```

The monitor waits for the architecture-control queue and authorized weights,
without allocating GPUs. Missing weights leave a visible waiting status. An
interrupted predecessor is reported as an error. The existing Torch 2.4.1
environment can run the official backbone; no package upgrades are required.
CPU tests use randomized official weights only to validate shapes, freezing,
and gradients, and do not produce formal feature caches or results. A pretrained
checkpoint has now passed strict loading and the CPU forward smoke test.

Results: `outputs/<classification|regression>/dinov3_vits16_frozen/`.
Log: `logs/run.log`; per-head logs are under `runs/<target>/train.log`.

## Current Main Regression Comparison (24h)

Run the ten current main regression tasks (including directly reported HCT and
eGFR) against `exp2_face_pretrained_head32_regression/outputs/20frame_face224`:

```bash
bash study/exp2_face_dinov3_frozen/launch_main_regression_screen.sh
screen -r exp2_dinov3_main24h_regression
```

This variant copies the exact main task CSVs and train-only scalers without a
new split search, uses the same selected native224 frame packets (including
explicit equivalence checks against the HCT/eGFR index), and retains all five
views. Every full batch contains twelve distinct lab events, twenty frames
per event, and one view per event. Unlike the earlier 12h run, it computes 240
individual frame SmoothL1 losses, not twelve pooled-view losses. Validation
loss is also frame-level; metrics use mean predictions over twenty original
frames. Initialization seeds match the corresponding main task runs.

The DINO encoder remains completely frozen and the original single-stage
head settings remain lr=2e-4, cosine minimum=1e-6, max epochs=80, patience=12,
weight decay=1e-3 and dropout=0.25. The comparator trains EN-B0 in two stages;
this is a representation/training-strategy comparison, not a backbone-only
ablation. No existing 12h DINO results or EfficientNet results are overwritten.

Outputs: `outputs/main24h_frame_loss/regression/dinov3_vits16_frozen/`.
Log: `logs/main24h_regression/run.log`.
Cache: `cache/main24h_frame_loss/`; only CLS vectors are stored, no copied pixels.
The launcher runs one small real-feature smoke test and then four-GPU head
training. At completion, training curves, prediction scatter plots with linear
fits, and paired MAE/RMSE/Pearson r/R2/explained-variance bar charts are generated
automatically. `paired_test_audit.csv` proves the test cohorts and labels agree;
`paired_baseline_comparison.csv` contains all test metrics.

### Head64 Variant

```bash
bash study/exp2_face_dinov3_frozen/launch_main_regression_screen.sh 64
screen -r exp2_dinov3_main24h_regression_head64
```

Only the hidden width changes to 64 (24,833 trainable head parameters). All
ten tasks, clinical labels, splits, seeds, views, batches, frame losses and
optimization settings are unchanged. The same frozen CLS cache is reused.
Head32 outputs remain untouched. Head64 outputs are under
`outputs/main24h_frame_loss_head64/regression/dinov3_vits16_frozen/`, with logs
in `logs/main24h_regression_head64/run.log`. Automatic metric comparisons show
EfficientNet-B0, DINO head32 and DINO head64 on exactly the same test videos.

## Frozen EfficientNet Controls And Five-Model Comparisons

The companion head32/head64 EfficientNet controls keep the entire ImageNet
backbone (including BatchNorm statistics) fixed and use exactly the DINO
80-epoch single-stage schedule, patience=12, lr=2e-4, cosine floor=1e-6,
weight decay=1e-3 and dropout=0.25. Heads consume 1280-dimensional GAP features
and contain 41,089/82,177 parameters. Clinical cohorts, frame losses, views,
seeds and evaluation are identical to the corresponding DINO runs. No feature
standardization or extra projection is fitted. One shared 717 MB EN cache
serves both the single-face and paired-face comparisons; frame packet equality
is checked before remapping the single-face data to the common cache.

```bash
bash study/common/launch_frozen_en_regression_controls_screen.sh
screen -r frozen_en_exp2_exp6_regression
```

This queue runs only the 42 new frozen-EN heads, never the completed DINO or
fine-tuned EN baselines. Exp2 results are under `outputs/frozen_en_b0/head32/`
and `outputs/frozen_en_b0/head64/` (then `regression/efficientnet_b0_frozen`).
Five-model comparison PNG/PDF files are `outputs/figures/five_model_*`, with
values in `outputs/five_model_test_comparison.csv` and strict test-identity
audit in `outputs/five_model_test_audit.csv`. Existing comparison figures
remain untouched. Common queue log: `study/common/logs/frozen_en_regression_controls/run.log`.
