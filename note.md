# Notes

## Overall
- Data quality
  * Signal polarity: negative in mirrors 1, 2, 4, 5, 6.
  * Mirror1 final update: 20251009, patient_id > 315.
  * TODO: verify whether all-zero windows are correctly excluded during auto wash.

## Experiments Before 2026-04 (Compact)

### Experiment 01: ECG+rPPG to BP Estimation
- Introduction
  * Goal: end-to-end blood pressure estimation from paired ECG/rPPG windows with patient-level split.
- Latest structure information
  * CNN + sequence modeling pipeline.
  * Latest tracked variant: Experiment 01T uses dual-branch CNN encoder (ECG/rPPG) + Transformer encoder + regression head (SBP/DBP).
- Latest result
  * Experiment 01 series remained validation-limited with frequent overfitting.
  * Experiment 01T status: implementation complete, local validation and CLI checks complete, full benchmark run still pending.

### Experiment 02: Cross-modal ECG<->rPPG Translation (GAN)
- Introduction
  * Goal: unsupervised bidirectional signal translation between ECG and rPPG.
- Latest structure information
  * Two generators (ECG->rPPG, rPPG->ECG) + two patch discriminators.
  * Generator: 1D encoder-residual-decoder; discriminator: 1D patch CNN.
  * Training loss: adversarial + paired L1 + cycle L1 + identity L1.
- Latest result
  * Refined pipeline runs successfully after removing BP-label filtering from the dataloader.
  * Quick smoke baseline (`light`, capped data): Val_pair=1.5329, Val_cycle=1.1251, combined=2.6580.

### Experiment 03: Masked Reconstruction (Joint ECG+rPPG)
- Introduction
  * Old target (rPPG->HR+SpO2) was deprecated due to label reliability and task mismatch.
  * Reframed as self-supervised masked reconstruction of joint ECG+rPPG.
- Latest structure information
  * Mask-aware reconstruction model with quality-weighted training.
  * Final pre-2026-04 direction selected from comparison: ECG-focus moderate emphasis.
  * Selected setting: `ecg_point_weight=1.25`, `grad_loss_weight=0.1`, `ecg_fft_loss_weight=0.02`.
- Latest result
  * Best balance from evaluated approaches: weighted_loss=0.107069, ECG_MAE=0.460197, rPPG_MAE=0.240319.
  * Combined ECG+rPPG MAE sum=0.700517 (best among compared pre-2026-04 approaches).

### Experiment 03-X: Candidate Structure Screening
- Introduction
  * Goal: screen high-potential architectures for Exp3 improvement.
- Latest structure information
  * Evaluated: `unet_gated`, `dual_head`, `tcn_ssm`, `cross_attention`.
- Latest result
  * Best model: `unet_gated` with WeightedLoss=0.169, ECG_MAE=0.431, rPPG_MAE=0.229.

### Experiment 03-1: ECG SQI Method Comparison
- Introduction
  * Goal: compare ECG signal quality indices and validate quality proxy choice.
- Latest structure information
  * Compared frequency-based SNR and non-frequency SQI features (template correlation, autocorrelation, morphology stability, artifact penalty, composite score).
- Latest result
  * Legacy frequency SNR showed weak relation to ECG quality cues.
  * Composite non-frequency SQI showed strong internal consistency and was preferred for Exp3 weighting.

### Experiment 04: Autoencoder-based Artifact/SQI Direction
- Introduction
  * Goal: replace fixed-rule quality filtering with reconstruction-based quality modeling.
- Latest structure information
  * Exp4: reconstruction error as quality signal with SNR-linked analysis.
  * Exp4-X: direct SQI regression from full data, with three models (`exp4-1`, `exp4-2`, `exp4-3`).
- Latest result
  * Full-data Exp4-X benchmark winner: `exp4-2`.
  * `exp4-2`: Val MAE=0.065481, Pearson=0.943353, Corr(Pred,SNR)=0.927810.

## Experiments On/After 2026-04 (Key Details)

## 2026-04-23
### Experiment 03 Split Update (ECG-only / rPPG-only)
- Target
  * Split Exp3 into two independent single-signal tasks:
    * `exp3_ecg`: ECG-only masked reconstruction.
    * `exp3_rppg`: rPPG-only masked reconstruction.
  * Keep `target_length=256` and use one contiguous masked window per segment.

- Implementation
  * Shared core modules:
    * `train/exp3_common/single_recon_dataloader.py`
    * `train/exp3_common/single_recon_model.py`
    * `train/exp3_common/single_recon_train.py`
    * `train/exp3_common/single_recon_visualize.py`
  * Task-specific entrypoints:
    * `train/exp3_ecg/*`
    * `train/exp3_rppg/*`

- Model and training design
  * U-Net style 1D reconstruction (both `light` and `full`) with skip connections for temporal alignment.
  * Mask-aware input: concatenated masked signal and visible mask.
  * Compound masked loss: MAE + gradient loss + FFT loss.
  * Quality weighting:
    * ECG uses ranked autocorrelation SQI.
    * rPPG uses ranked SNR SQI.

- Outputs
  * Checkpoints:
    * `train/checkpoints/exp3_ecg_<variant>_best.pt`
    * `train/checkpoints/exp3_ecg_<variant>_final.pt`
    * `train/checkpoints/exp3_rppg_<variant>_best.pt`
    * `train/checkpoints/exp3_rppg_<variant>_final.pt`
  * Plots:
    * `train/exp3_ecg/plots/*`
    * `train/exp3_rppg/plots/*`

