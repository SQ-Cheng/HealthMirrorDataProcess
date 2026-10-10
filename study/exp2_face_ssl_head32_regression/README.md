# SSL-Initialized Independent Face Regression

This independent experiment uses **BYOL**, not UM-MAE. It retains the main
ImageNet-initialized EfficientNet-B0 and its head32 raw-value regression architecture.
There are three stages: one label-free backbone pretraining stage, then ten
independent frozen-head/full-backbone two-stage regression runs. Downstream models
start from the same exported SSL features, but do not share trainable parameters.

## Patients and Leakage

The main experiment's ten per-task patient splits are incompatible with a common
pretraining encoder. Search 512 common patient-split candidates, using the inherited
raw-value/abnormal-score KS, normalized Wasserstein, cohort-size and label-rate
distribution objectives for every task. Candidate152 passes all sixty distribution
pairs. The common train/validation/test patient counts are **339/112/112**.

SSL includes only patients assigned to training, including their other validated
same-admission sessions without a nearby task label. Validation/test/unknown patients
are excluded from every SSL forward pass, normalization update and EMA update.
Split stratification uses labels, as requested; no clinical labels or timestamps
are supplied to the BYOL learner. Each downstream scaler uses only its task's
training rows. Source times, matching and raw labels remain the exact current-main
24h records.

The SSL pool contains **996 videos, 339 patients and 1,275,659 accepted native224
frames**. All five views are used as anchors once for every frame each epoch:
**6,378,295 real anchors per epoch**. No twenty-frame subsampling is applied in SSL.
The all-frame packet index is approximately10MB; original lossless FFV1 videos
are reused without copied pixels or image caches.

## BYOL

- Online encoder: EfficientNet-B0 features and spatial average pooling (1280).
- Projector: FC(1280,2048), BatchNorm, ReLU, FC(2048,256).
- Predictor: FC(256,2048), BatchNorm, ReLU, FC(2048,256).
- Target: EMA copy of the encoder/projector, no predictor and no gradient.
- Loss: mean of the two symmetric normalized-MSE prediction/stop-gradient target
  directions. No negative samples, labels or label-aware positive selection.
- Positives: the same accepted frame under two different existing view types.
- Views: original, horizontal flip, center crop, brightness, contrast. Partners
  rotate among the other four types across epochs. No extra grayscale/blur policy.
- Encoder lr1e-4; projector/predictor lr1e-3; AdamW weight decay1e-4; gradient clip1.
- Eight fixed epochs, two-epoch linear warmup then cosine floor1e-6.
- Target parameter EMA starts at0.996 and approaches1; teacher BatchNorm buffers
  use their own local training-batch updates, not held-out data.
- Use the final online encoder, without validation/test-based SSL selection.
  Monitor projection variance to block an obviously collapsed representation.

Four GPUs use one process each under DDP, **64 anchors per GPU / 256 global anchors**.
Global batches first maximize distinct videos and are shuffled before rank slicing.
When few videos remain, fill with other unused frame/view units from those videos.
Every real unit is retained exactly once. Empty ranks on the final tail use a
training-only loss-zero forward slot; global loss normalization and diagnostics
exclude these slots. DDP per-rank BN state is updated independently and rank0
buffers are exported, with random global-batch rank allocation.
SSL runs eagerly to avoid CUDA Graph private-pool growth and is isolated in a
subprocess so its CUDA state is freed before downstream scheduling.

Method sources: [BYOL paper](https://arxiv.org/abs/2006.07733) and
[official implementation](https://github.com/google-deepmind/deepmind-research/tree/master/byol).
This adapts the original recipe to ImageNet EfficientNet-B0, AdamW and the existing
five-view policy; it does not claim to reproduce the paper's ImageNet hyperparameters.

## Downstream

Ten original targets are retained, including direct HCT and creatinine-method eGFR.
Each task uses twenty nonadjacent native224 frames per video and all five views.
Full batches contain twelve distinct matched reports, twenty frames and one view
per report:240 frames. Targets are train-only median/IQR scaled raw values with
unweighted frame SmoothL1(beta0.5). Head training uses lr2e-4, forty epochs,
patience10; full fine-tuning uses lr1e-5, sixty epochs, patience12. Remaining
optimizer/scheduler/selection settings match the main.

Downstream jobs dynamically use four GPUs. Inductor compilation is retained with
dynamic batch shapes and CUDA Graphs disabled. SSL optimizer/decoder heads are
not carried into the regressors. Every downstream checkpoint records the hash of
the identical initial SSL encoder. Histories, raw-unit metrics, predictions and
figures are saved per task and aggregated. Comparisons report separate full test
cohorts and common held-out test videos; original main splits differ, so this is
not presented as a matched-split independent control.

## Entry

```bash
# Rebuild split and frame index before any training:
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp2_face_ssl_head32_regression.run_all --prepare-only
# Real four-GPU BYOL update and temporary two-stage Hb transfer:
HEALTHMIRROR_FACE_SOURCE=face224 /root/miniconda3/envs/healthmirrorenv/bin/python -u \
  -m study.exp2_face_ssl_head32_regression.run_all --reuse-prepared --smoke
# Complete three-stage training in detached screen:
bash study/exp2_face_ssl_head32_regression/launch_screen.sh
```

Screen: `exp2_ssl_byol`. Log: `logs/run.log`. Re-launch resumes SSL at the
last completed epoch and skips matching completed downstream models. SSL records
optimizer/scaler, each rank's RNG, dataset contracts and feature weights. No
unrelated main results are overwritten. Final plots are generated automatically.
Use `--phase ssl` or `--phase downstream` on the screen launcher to run a single
part. Restored SSL requires the original GPU count and unchanged prepared contract.

## Explicit Early Transition

On request, BYOL was interrupted during epoch3. The last saved checkpoint contains
two completed epochs / 49,832 batches, not the unsaved epoch3 updates. Its online
feature tensors were exported to `outputs/ssl/encoder.pt` after checking their
contract, variance audit, finiteness and exact EfficientNet-B0 state-dict structure.
`outputs/ssl/transition.json` records actual versus planned epochs and both
checkpoint/encoder hashes. The original SSL checkpoint, histories and prepared
split/scalers are preserved; the original experiment manifest is not rewritten,
so its data contract remains valid.

```bash
bash study/exp2_face_ssl_head32_regression/launch_screen.sh --phase downstream
```

An explicitly accepted early encoder is also skipped by later full-pipeline
relaunches, so they do not silently restart BYOL. Downstream initialization always
checks the exported encoder's recorded hash and patient scope.
