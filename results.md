# Experiment Results Report

## Experiment 01: ECG+rPPG -> BP

### Model structure (Exp01 family)

- Core model (LSCN): dual 1D CNN branches (ECG and rPPG), each with multi-scale convolutions, followed by feature fusion, single-layer LSTM, and a linear SBP/DBP head.
- Main structural changes across Exp01-01 to Exp01-13: kernel size tuning, dropout tuning, LSTM hidden-size changes, downsampling/stride changes, and reduced input size for compact variants.

### Exp01 summary table 

**Exp01-01 to 01-09 are contaminated by faulty data**

| Experiment | SBPVa (mean +- SD) | Simple comment |
|---|---:|---|
| Exp01-01 | 0.05 +- 14.393 | Baseline; possible overfitting |
| Exp01-02 | -1.72 +- 16.610 | Dropout 0.2; still overfitting |
| Exp01-03 | -0.43 +- 14.098 | Dropout 0.3; overfitting concern remains |
| Exp01-04 | -6.08 +- 17.786 | Patient-level split + dropout 0.5; unstable and degraded |
| Exp01-05 | N/A | Structure fix (removed redundant maxpool, stride-2 conv); no reliable full-result metric recorded |
| Exp01-06 | -5.45 +- 18.543 | LSTM hidden size reduced to 64; likely overfitting |
| Exp01-07 | -4.93 +- 18.419 | Feature extraction redesign; likely overfitting |
| Exp01-08 | around -4 +- 18 | Kernel size changed to 49/15; overfitting and seed sensitivity |
| Exp01-09 | N/A | Faulty-data fix test; performance close to mean-guessing |
| Exp01-10 | -2.07 +- 16.933 (LR 1e-4); -3.51 +- 17.459 (LR 1e-3) | Kernel 25/9, dropout 0.5; training fits better than validation |
| Exp01-11 | -5.55 +- 17.515 | Smaller model (47,618 params), Huber loss; overfitting |
| Exp01-12 | N/A | Window reduced to 2 s; full-result metric not recorded |
| Exp01-13 | N/A | CNN stride set to 1; full-result metric not recorded |

## Experiment 01T: CNN + Transformer BP prediction

### Model structure

- Dual 1D CNN encoders extract ECG and rPPG features separately.
- Features are concatenated into token sequences, projected to transformer dimension, then processed by a multi-layer transformer encoder.
- A class token output is passed to an MLP head to regress SBP and DBP.

- **Worse than LSCN**

## Experiment 01I: CNN + LSTM + Image input

- Building model.

## Experiment 02: ECG <-> rPPG translation

### Model structure

- Bidirectional GAN with two generators and two discriminators.
- Generator: 1D encoder -> residual blocks -> transposed-conv decoder for cross-modal waveform translation.
- Discriminator: patch-style 1D CNN classifier over local temporal segments.
- Variants: light (fewer channels/residual blocks) and full (higher capacity).

- Results not availiable yet - Training not conducted due to local performance limit.

## Experiment 03: Joint ECG+rPPG masked reconstruction

### Model structure

- Input is masked ECG+rPPG plus visible-mask channels.
- Light variant: compact conv encoder with residual blocks and transposed-conv decoder.
- Full variant: deeper stem with multi-dilation residual body and deeper decoder.
- Loss-level model variants in full training: baseline, ECG-plus (stronger ECG emphasis), ECG-focus moderate (balanced ECG-detail emphasis).

### Full-training optimization comparison

| Approach | Checkpoint | Weighted loss | ECG MAE | rPPG MAE | ECG+rPPG MAE sum |
|---|---|---:|---:|---:|---:|
| Baseline full reconstruction | exp3_full_best.pt | 0.106877 | 0.462225 | 0.238847 | 0.701072 |
| ECG-plus strong emphasis | exp3_full_ecgplus_best.pt | 0.107773 | 0.461137 | 0.242826 | 0.703964 |
| ECG-focus moderate (final) | exp3_full_ecgfocus_best.pt | 0.107069 | 0.460197 | 0.240319 | 0.700517 |

- Selected default: ECG-focus moderate emphasis.

### Exp03-1 SQI comparison study

### SQI model structure

- Non-frequency ECG SQI combines template correlation, autocorrelation, morphology stability, and artifact penalty into a composite score.
- Legacy frequency SNR is treated as a separate baseline proxy for comparison.

- Full-dataset pool: 12,240 windows (2,345 patients).
- Analysis sampled 800 windows.
- Main finding: non-frequency ECG SQI features are more coherent than legacy frequency-SNR proxy.

## Experiment 03X: Alternative model structures

### Model structure summary

- unet_gated: mask-aware U-Net encoder-decoder with gated skip fusion.
- dual_head: shared encoder with separate ECG and rPPG decoder heads.
- tcn_ssm: dilated temporal CNN with gated residual/state-mixing blocks.
- cross_attention: temporal self-attention blocks plus residual refinement.
- mamba: Mamba-style token-mixing stack with modality-specific output heads (implemented, but not included in the full-training metrics table below).

### Full-training evaluation metrics

Source: train/exp3x/plots/*_metrics.json

| Model | Best epoch | Mask ratio | Weighted loss | ECG MAE | rPPG MAE | ECG corr | rPPG corr |
|---|---:|---:|---:|---:|---:|---:|---:|
| dual_head | 55 | 0.3 | 0.182156 | 0.437045 | 0.232855 | 0.695388 | 0.935080 |
| tcn_ssm | 59 | 0.3 | 0.187203 | 0.472316 | 0.216805 | 0.635540 | 0.939983 |
| cross_attention | 59 | 0.3 | 0.195686 | 0.477225 | 0.235178 | 0.612133 | 0.932183 |
| unet_gated | 59 | 0.5 | 0.203469 | 0.470560 | 0.253030 | 0.644785 | 0.922290 |

- Best weighted loss in exported evaluation: dual_head.

## Experiment 04: Unsupervised artifact detector (autoencoder SQI)

### Model structure

- 1D convolutional autoencoder with explicit bottleneck latent channels.
- Encoder progressively downsamples temporal features; decoder reconstructs waveform via transposed convolutions.
- Variants: light and full differ in channel width, latent dropout, and temporal downsample factor.

- Results not availiable yet - Training not conducted

## Experiment 04X: Full-data SNR-ranked SQI regression

### Model structure

- exp4-1: compact pure CNN regressor with global pooling head.
- exp4-2: CNN feature stem followed by bidirectional GRU temporal regression head.
- exp4-3: multi-scale dilated temporal CNN regressor.

### Full-data benchmark (full training)

Dataset: 12,240 windows (2,345 patients), split 9,018 / 3,222.

| Model | Val MAE | Pearson | Corr(Pred,SNR) |
|---|---:|---:|---:|
| exp4-1 | 0.101782 | 0.877300 | 0.868098 |
| exp4-2 | 0.065481 | 0.943353 | 0.927810 |
| exp4-3 | 0.096558 | 0.891255 | 0.882799 |

- Best overall: exp4-2.

## High-level summary

- Exp01 family: repeated overfitting/instability on validation.
- Most reliable full-training results in current records:
  - Exp03 ECG-focus moderate model for joint reconstruction balance.
  - Exp4X exp4-2 for SQI regression.
