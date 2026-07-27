# Exp2 All-Frame Multi-Output Abnormal-Score Regression

Canonical ID: `exp2_regression_allframes_head32_multitask`.

This experiment trains one model per pretrained architecture to predict all five
abnormal scores simultaneously.

## Model and Training

- Backbones: unchanged ImageNet-pretrained MobileNetV3-Small and EfficientNet-B0
- Head: Linear -> LayerNorm -> SiLU -> Dropout -> Linear(5), hidden width 32
- Output order: `hemoglobin_low`, `pco2_low`, `po2_low`,
  `high_blood_pressure`, `lactate_high`
- Stage 1: freeze the backbone and train the head at `2e-4`
- Stage 2: unfreeze all parameters and fine-tune at `1e-5`
- Early stopping and stage selection: macro video-level validation MAE

Every decodable source frame is streamed from the existing compact byte-offset
index. Training uses five deterministic views: original, horizontal flip, 90%
center crop, brightness +6%, and contrast +8%. Validation and test use the original
view only. Frames are resized to 224 x 224 and ImageNet-normalized on the GPU.

## Partial Labels

Per-target records are cleaned before they are joined. A video is excluded only
from a target for which it contains both normal and abnormal events; valid labels
for its other targets remain usable. For each clean target/video pair, the closest
lab event within the corrected 24-hour window supplies the abnormal score.

The five record sets are outer-joined by video. A video is retained when at least
one target is available. Missing targets are represented by a binary mask and do
not enter the loss. This retains all 673 usable videos; requiring all five labels
would retain only 46.

One global patient-level 60/20/20 split is shared by all outputs. The deterministic
2,048-candidate search checks every task's raw-value and abnormal-score
distributions, split sizes, and normal/abnormal coverage. This prevents the same
patient from appearing in different splits through different tasks.

Training minimizes masked SmoothL1 loss (`beta=0.5`). Each task is weighted by
`mean(training frame-label counts) / task training frame-label count`, using actual
source-frame counts. This gives each output equal aggregate influence without
adding abnormal-side weights or fabricating missing labels. Exact counts and
weights are written to `outputs/task_loss_weights.json`.

## Outputs

The experiment writes machine-readable split, conflict, distribution, target-mask,
loss-weight, and experiment manifests. Each architecture saves:

- `history.csv` and `history.png`
- per-task and macro train/validation/test metrics
- long-form video predictions and compressed masked frame predictions
- head-stage, fine-tune-stage, and selected checkpoints

Decoded images are never cached on disk.

## Start

Run preparation without training:

```bash
python -m study.exp2_face_pretrained_allframes_head32_multitask_regression.run_all \
  --prepare-only
```

Start the formal experiment in a detached screen:

```bash
bash study/exp2_face_pretrained_allframes_head32_multitask_regression/launch_screen.sh
```

Use `--overwrite` only when intentionally replacing an existing run.
