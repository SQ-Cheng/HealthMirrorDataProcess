# Exp2 Native Video + 512 Hz ECG Mamba Regression

This experiment trains one independent abnormal-score regressor for each of:

- `hemoglobin_low`
- `pco2_low`
- `po2_low`
- `high_blood_pressure`
- `lactate_high`

It uses the official `state-spaces/mamba` selective state-space implementation.
No previous Mamba code in this repository is imported or reused.

## Regression Target

For each measurement, the signed distance from its clinical boundary is divided
by a target-specific scale and transformed with `asinh`:

```text
abnormal_score = asinh(signed_distance_from_boundary / clinical_scale)
```

Scores below zero are on the normal side, zero is the clinical boundary, and
scores above zero are on the abnormal side. The exact per-target thresholds and
scales are saved to `score_definition.json`. Each target has a separate model.

## Data Contract

Labels come from the corrected Exp2 source tables. Lab measurements must be
within the corrected 24-hour video window, timestamps must use `Asia/Shanghai`,
and videos with conflicting positive and negative events for one target are
excluded from that target. Each target receives its own distribution-balanced,
patient-disjoint 60/20/20 split.

For every usable raw recording:

1. `video.avi.ts` and `ecg_log.csv` must contain finite, strictly increasing Unix
   timestamps.
2. The exact temporal intersection between RGB video and ECG is computed.
3. The overlap is divided into non-overlapping 8-second windows.
4. Frame and ECG boundaries are found from timestamps, not nominal rates.
5. A window is excluded if any source ECG timestamp gap used for interpolation
   exceeds 60 ms.

Every recorded RGB frame remains at `3 x 128 x 128`; there is no video
interpolation or temporal frame sampling.

ECG is linearly interpolated from its real, irregular Unix timestamps onto the
uniform grid `window_start + arange(4096) / 512`. The grid is exactly 512 Hz and
8 seconds long. Extrapolation is forbidden. Resampling is followed by per-window
robust amplitude normalization. The model receives one ECG channel containing
only amplitude; sampling intervals are not an input channel.

## Model

```text
Native RGB video: (Tv, 3, 128, 128)
  -> shared lightweight 2D frame encoder
  -> one 96-dimensional token for every recorded frame

Uniform ECG: (4096, normalized amplitude)
  -> learned Conv1D tokenizer (total stride 16)
  -> 256 x 96-dimensional local morphology tokens

Video and ECG tokens
  -> add modality embedding and continuous relative-time embedding
  -> stable merge by acquisition time
  -> append learned summary token
  -> 4 x [RMSNorm -> official Mamba(d_model=96, d_state=16,
                                   d_conv=4, expand=2) -> residual]
  -> RMSNorm
  -> Linear(96 -> 32) -> LayerNorm -> SiLU -> Dropout -> Linear(32 -> 1)
  -> predicted abnormal score
```

The ECG convolution stride is learned model tokenization after uniform
resampling. Mamba parameters remain FP32; training uses CUDA FP16 autocast.

## Training and Evaluation

- Objective: unweighted `SmoothL1Loss(beta=0.5)` on abnormal score
- Auxiliary classification or abnormal-side loss: none
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- Schedule: cosine decay to `1e-6`
- Train/evaluation batch size: `24`
- Maximum: 40 epochs
- Early stopping: 8 epochs on video-level validation MAE
- Training augmentation: one temporally consistent horizontal flip
- Evaluation: average all window predictions per video

Every epoch records train/validation loss, MAE, RMSE, Pearson correlation,
Spearman correlation, sign bACC/AUC, throughput, learning rate, and peak GPU
memory. Sign metrics are diagnostic only and never contribute to loss or
checkpoint selection.

## Dependencies

The validated environment uses PyTorch `2.4.1+cu121` and the official
`mamba-ssm==2.2.6.post3`.

```bash
bash study/exp2_video_ecg_mamba/install_mamba.sh
```

## Commands

Prepare and audit data without training:

```bash
python -m study.exp2_video_ecg_mamba.run_all --prepare-only
```

Run a bounded smoke test:

```bash
python -m study.exp2_video_ecg_mamba.run_all \
  --targets hemoglobin_low \
  --smoke-test \
  --output-dir study/exp2_video_ecg_mamba/outputs_smoke \
  --overwrite
```

Formal detached launch:

```bash
bash study/exp2_video_ecg_mamba/launch_screen.sh
```
