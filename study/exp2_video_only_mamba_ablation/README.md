# Exp2 Video-Only Mamba Controlled Ablation

This experiment is paired with `study/exp2_video_ecg_mamba`. It measures the
effect of removing ECG while keeping the native-video Mamba regression pipeline
otherwise unchanged.

## Changed Variable

```text
Parent:   native video frame tokens + uniformly resampled ECG tokens -> Mamba
Ablation: native video frame tokens only                           -> Mamba
```

The video-only Dataset never opens or decodes `ecg_log.csv`. ECG values,
timestamps, and tokens are absent from the forward pass.

## Controlled Variables

- The exact parent window rows, row order, window IDs, video frame boundaries,
  target scores, and patient-level train/validation/test assignments are copied
  byte-for-byte and SHA256 recorded.
- The same five independent abnormal-score regression tasks are trained.
- Native RGB input remains `3 x 128 x 128`; every frame in each 8-second window
  is used without interpolation or temporal sampling.
- The frame encoder, continuous time embedding, four-layer official Mamba
  backbone, summary token, and 32-dimensional head are unchanged.
- The same base seed and parent job-seed derivation are used.
- Shared modules are constructed in the same order, so their initial tensors
  are identical under the paired job seed.
- Batch sizes, workers, horizontal-flip augmentation, SmoothL1 loss, optimizer,
  learning rate, schedule, epoch limits, early stopping, evaluation aggregation,
  and checkpoint criterion are unchanged.

The parent ECG tokenizer is retained only as a frozen initialization placeholder.
It is never called and is excluded from optimization. This avoids changing the
random initialization of later shared modules.

## Parent Completion Contract

A formal run requires all five rows in the parent `run_index.csv` to have
`status=ok`. It snapshots the parent data manifests and exact window CSVs before
training. `--allow-incomplete-parent` exists only for bounded preflight tests.

## Commands

Formal detached launch:

```bash
bash study/exp2_video_only_mamba_ablation/launch_screen.sh
```

Monitor the paired experiment and launch automatically after successful
completion:

```bash
bash study/exp2_video_only_mamba_ablation/monitor_then_launch.sh
```

Bounded smoke test while the parent is still running:

```bash
python -m study.exp2_video_only_mamba_ablation.run_all \
  --targets hemoglobin_low \
  --smoke-test \
  --allow-incomplete-parent \
  --output-dir study/exp2_video_only_mamba_ablation/outputs_smoke \
  --overwrite
```
