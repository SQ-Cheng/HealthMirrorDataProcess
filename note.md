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
  * Baseline encoder-decoder reconstruction (pre-U-Net version) is now the active default after rollback.
  * Mask-aware input: concatenated masked signal and visible mask.
  * Baseline weighted masked SmoothL1 loss is used in training.
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

### Experiment 03 Rollback Update (Baseline Recovery)
- Background
  * Re-training after the U-Net/extra-loss modification showed worse quality than the prior baseline.

- Rollback actions
  * Restored `train/exp3_common/single_recon_model.py` to the original single-recon baseline architecture:
    * `light`: stride-2 encoder + residual body + transposed-conv decoder.
    * `full`: `stem` + 3 residual blocks + transposed-conv decoder.
  * Restored `train/exp3_common/single_recon_train.py` loss path to baseline weighted masked SmoothL1 (removed gradient/FFT terms from default training flow).

- Additional helpful improvements
  * Improved checkpoint robustness in `train/exp3_common/single_recon_visualize.py`:
    * Visualization now tries candidate checkpoints (`best`, then `final`) and loads the first one compatible with current model keys.
    * This avoids breakage when checkpoint files come from mixed architecture generations.
  * Added `model_family="single_recon_v1"` metadata to newly saved checkpoints for clearer future compatibility tracking.

- Verification (local smoke)
  * Visualization smoke (full variant) completed successfully and saved plot:
    * `train/exp3_ecg/plots/exp3_ecg_full_masked_recon.png`
    * Masked MAE: `0.140958`
  * Training smoke (full variant, 1 epoch capped subset) completed successfully:
    * Best val loss: `0.2573`
    * History: `train/exp3_ecg/plots/exp3_ecg_full_rollback_smoke_history.csv`

- Recommended usage after rollback
  * Use explicit tags for new rollback runs to avoid mixed-generation checkpoint confusion:
    * `python train/exp3_ecg/exp3_ecg_train.py --variant full --checkpoint-tag _legacyv1`
    * `python train/exp3_rppg/exp3_rppg_train.py --variant full --checkpoint-tag _legacyv1`

## 2026-05-11
### Lactate Association Analysis (思路1)
- **Objective**: Find relationships between lactate values and other features (BP, age, gender, HR, SpO2, RR, temperature).
- **Data**: 814 lactate records from 245 unique patients across mirrors 1,2,4,5,6.
- **Lactate stats**: mean=1.99±0.47 mmol/L, median=1.94, range=[1.02, 5.45].
- **Visualizations**: 9 plots saved to `lactate_analysis/` directory.

#### Key Results
| Feature | vs Lactate Mean | p-value | Interpretation |
|---------|----------------|---------|----------------|
| DBP | r=0.098 | 0.025 | Very weak positive (barely significant) |
| SBP | r=0.010 | 0.827 | No correlation |
| Age | r=0.042 | 0.230 | No correlation |
| Gender | M:2.01, F:1.94 | 0.152 | No significant difference |
| Heart Rate | r=0.119 | 0.001 | Weak positive (significant) |
| SpO2 | r=0.021 | 0.574 | No correlation |
| Respiratory Rate | r=0.038 | 0.358 | No correlation |
| Temperature | r=0.114 | 0.359 | No correlation (n=66) |

#### Threshold Analysis (High vs Normal Lactate)
- No threshold (1.0-3.0 mmol/L) showed significant BP differences.
- Best (still non-significant): >=1.5 mmol/L → DBP +2.43 mmHg (p=0.118).

#### Conclusion
- **Lactate has very weak to no correlation** with all available features.
- The strongest (still weak) signal: **lactate_mean vs HR** (r=0.119, p=0.001) and **lactate_mean vs DBP** (r=0.098, p=0.025).
- These correlations are too weak for predictive modeling.
- Possible reasons: (1) Lactate is a time-varying acute marker, but our data uses per-patient aggregates (mean/min/max). (2) The population may be generally healthy with normal lactate ranges. (3) Lab measurements and vital signs may not be temporally aligned.

## 2026-04-26
### Experiment 03-TCN
- Baseline: Best val loss: 0.1506
- Ablation 01
  * Delete `silu` in `dialatedresblock`
  Best val loss: 0.1533
- Ablation 02
  * Change `nn.Dropout` to `nn.Dropout1d`
  Best val loss: 0.1536
 - Ablation 03 *wating to be fixed: larger receptive field*
  * targht-length 512, mask-ratio 0.2
  Best val loss: 0.1980
- Ablation 04


### Experiment 03-Transformer
- Baseline: Best val loss: 0.2196

### Experiment 03-mamba
- waiting to be modified to real mamba
- baseline: Best val loss: 0.2279

### Experiment 03-GAN



## 2026-05-12
### Temporal Lactate Analysis (Δ-Lactate)
- **Objective**: Use temporal alignment of individual lactate measurements with recording sessions to detect Δ-lactate ↔ Δ-vital relationships.

#### Data & Matching
- Lactate measurements parsed: 5152 from 329 unique patients in XLSX
- Recording sessions extracted: 1403 from 441 patients (mirrors 1,2,4,5,6)
- Temporally matched sessions: 850 (60.6%)
  - Primary match (≤24h before): 426
  - Secondary match (≤7d): 394
  - Tertiary match (beyond 7d): 30

#### Δ-Feature Analysis
- Patients with ≥2 matched sessions: 209
- Consecutive session pairs: 598
- Mean Δ-lactate: 0.065 ± 0.881 mmol/L
- Δ-time between sessions: mean=2.3 days, median=1.1 days

#### Δ-Lactate vs Δ-Vital Correlations
| Δ-Vital | N (pairs) | Pearson r | p-value | Spearman ρ | Interpretation |
|---------|-----------|-----------|---------|------------|----------------|
| SBP | 20 | -0.543 | p=0.0134 | 0.017 | Strong |
| DBP | 20 | -0.363 | p=0.1160 | 0.083 | Not significant |
| HR | 8 | 0.056 | p=0.8947 | 0.125 | Not significant |
| SpO2 | 3 | — | — | — | Insufficient data |
| RR | 5 | 0.250 | p=0.6850 | 0.250 | Not significant |
| Temp | 13 | -0.377 | p=0.2047 | -0.121 | Not significant |

#### Key Findings
- Strongest signal: SBP (r=-0.543, p=0.0134) — Strong
- Significant correlations (p<0.05): 1/5
**Unusable. Influenced by probably error values**

#### Comparison with Static Analysis (2026-05-11)
- Static analysis found very weak correlations (best: lactate_mean vs HR r=0.119, lactate_mean vs DBP r=0.098)
- The temporal Δ-approach examines **within-patient changes over time**, removing inter-subject variability
- Temporal analysis reveals relationships not visible in static aggregate analysis, confirming the value of temporal alignment.

#### Output Files
- `lactate_analysis/delta_lactate_vs_delta_*` — scatter plots for each vital
- `lactate_analysis/delta_correlation_heatmap.png` — full Δ-feature correlation matrix
- `lactate_analysis/delta_subgroup_analysis.png` — Δ-lactate tertile comparison
- `lactate_analysis/match_type_summary.png` — quality of temporal matching
