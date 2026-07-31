# Exp2 20-Frame Head64 Abnormal-Score Regression

Canonical ID: `exp2_raw_video_20frame_head64_regression_balanced_split`.

This experiment is derived directly from
`study/exp2_face_pretrained_head32_regression`. It trains independent
MobileNetV3-Small and EfficientNet-B0 models for:

- `hemoglobin_low`
- `po2_low`
- `lactate_high`

Each video is matched to the nearest valid lab measurement within 24 hours. Twenty
deterministic non-adjacent RGB frames are selected from 5% through 95% of the raw
video. Training uses the original, horizontal flip, 90% center crop, brightness +6%,
and contrast +8% views; validation and test use only the original frames.

The target is the continuous abnormal score used by the reference regression:
negative is on the normal side, positive is on the abnormal side, and zero is the
clinical boundary. Models minimize unweighted SmoothL1 loss with `beta=0.5`.

The regression head is
`Linear -> LayerNorm -> SiLU -> Dropout -> Linear(1)` with hidden width 64. Stage 1
trains only this head at `2e-4`; stage 2 unfreezes the complete backbone and trains all
parameters at `1e-5`. Epoch limits, early stopping, batches, patient-disjoint balanced
split search, mixed precision, and four-GPU dynamic scheduling match the reference.

Training uses `torch.compile(mode="reduce-overhead")` for both stages, together with
AMP and channels-last execution.

Successful completion automatically writes all histories, metrics, predictions,
checkpoints, manifests, and four validated result figures under `outputs/20frame`.

## Start

```bash
bash study/exp2_face_pretrained_head64_regression/launch_screen.sh 20frame --overwrite
```

```bash
screen -r exp2_face_pretrained_head64_regression_20frame
```
