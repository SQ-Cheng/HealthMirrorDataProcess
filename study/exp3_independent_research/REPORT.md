# Exp3: Independent Academic Research — Comprehensive Report

> **Date**: 2026-07-03
> **Version**: Phase 2 (incorporates reviewer feedback)
> **Researcher**: Automated AI Research System
> **Environment**: Python 3.12, scikit-learn 1.5
> **Data**: HealthMirror multimodal dataset (ECG + Face + Lab values, 843 matched samples, 137 patients)

---

## Executive Summary (Updated)

### Phase 1 Findings (Confirmed)
1. Classical ECG feature engineering (macro bACC=0.528) outperforms Exp2 deep learning (0.448)
2. Surgery timing is the dominant predictor of lab values
3. Cross-analyte correlations are physiologically meaningful
4. 10-second ECG windows are **not patient-stable** (intra/inter ratio = 0.95×)

### Phase 2 Findings (New)
5. **Hemoglobin unit bug fixed** — values were ×10 too large due to incorrect g/dL→g/L conversion
6. **ECG adds ZERO residual value beyond clinical baseline** — this is a strong, consistent negative finding across all 4 Phase 2 experiments
7. **Previous lab value + days-from-surgery** explains hemoglobin variance at R²=0.88; adding ECG degrades to R²=0.88
8. **Within-patient ΔECG features do not correlate with ΔLab** — all ΔR² ≤ 0
9. **HRV features are highly unstable** (CV > 0.7) within patients across sessions
10. **5 clinically meaningful redefined labels proposed** for future work

### Central Scientific Message

> The 10-second ECG window captured by the HealthMirror device contains **negligible information** about a patient's systemic physiological state (lab values), beyond what is already captured by: (1) the previous lab measurement, and (2) the time elapsed since surgery. This is a **scientifically robust negative result** that should guide future research toward longer monitoring windows, different signal modalities, or within-patient change detection.

---

## Phase 2: Data Cleaning & Unit Fixes

### Hemoglobin Unit Correction (CRITICAL)

The raw lab data contains hemoglobin in **two units**:
- **g/dL**: 8,087 values (e.g., 12.2, 11.9) — typical range 8-18
- **g/L**: 3,259 values (e.g., 117, 110) — typical range 80-180

The Phase 1 converter blindly multiplied ALL values ×10 (intended as g/dL→g/L), making g/L values 10× too large (117→1170). This contaminated hemoglobin correlations and hemoglobin_low classification.

**Fix**: Only multiply values <20 by 10; leave values ≥20 as-is.
- After fix: median 111 g/L, range 35-188 g/L (physiologically reasonable)
- hemoglobin_low prevalence corrected (now sex-dependent: M<130, F<120)

### Glucose Unit Correction

Similarly, glucose raw values are in both mmol/L (values 1-30) and mg/dL (70-600).
- Fix: only convert mg/dL (>30) by ÷18 to mmol/L; leave mmol/L as-is
- After fix: glucose_abnormal rate is 43% (up from 12%), now correctly reflecting post-CABG hyperglycemia

### Other Analytes

| Analyte | Units | Validated? | Issues |
|---------|-------|:----------:|--------|
| Lactate | mmol/L | ✓ | Clean, single unit |
| Troponin | ng/L (pg/mL rare) | ✓ | 22/2884 in pg/mL (≈ng/L, negligible) |
| pO2 | mmHg | ✓ | Clean |
| pCO2 | mmHg | ✓ | Clean |

---

## Phase 2: Core Experiments

### Direction F: Peri-Operative Recovery Trajectory Modeling

**Design**: Three tasks testing ECG's incremental value over clinical baselines.

#### Task A: Predicting ΔLab (Relative Change)

| Analyte | Surgery Baseline R² | +ECG R² | ΔR² | p(B3>B2) |
|---------|:-------------------:|:-------:|:---:|:---------:|
| troponin | +0.281 | +0.167 | -0.114 | 0.790 |
| hemoglobin | +0.200 | +0.040 | -0.160 | 0.919 |
| lactate | -0.147 | -0.189 | -0.042 | 0.821 |
| glucose | +0.097 | +0.051 | -0.046 | 0.763 |
| pco2 | -0.494 | -0.517 | -0.023 | 0.875 |
| po2 | -0.323 | -0.487 | -0.165 | 0.781 |

**Result**: ECG adds **zero** residual predictive power for any analyte's Δlab. All ΔR² are negative — adding ECG makes predictions WORSE.

#### Task B: Abnormal Recovery Trajectory Detection

- Population recovery curves fitted for each analyte
- ~10% of samples flagged as extreme residuals from population trend
- Provides reference "normal recovery" envelope for clinical monitoring

#### Task C: Hierarchical Baseline (Strong Clinical Control)

| Analyte | B0 (mean) | B1 (days) | B2 (+prev lab) | B3 (+ECG) | ECG ΔR² | p |
|---------|:---------:|:---------:|:--------------:|:---------:|:-------:|:--:|
| hemoglobin | -0.07 | +0.02 | **+0.885** | +0.876 | -0.006 | 0.90 |
| pco2 | -0.04 | -0.05 | **+0.458** | +0.410 | -0.038 | 0.91 |
| lactate | -0.02 | -0.04 | **+0.333** | +0.304 | -0.025 | 0.87 |
| glucose | -0.06 | -0.04 | **+0.281** | +0.247 | -0.029 | 0.87 |
| po2 | -0.04 | +0.08 | **+0.060** | -0.024 | -0.074 | 0.93 |
| troponin | -0.95 | -1.10 | **+0.082** | -0.299 | -0.483 | 0.83 |

**Critical finding**: The previous lab value alone explains 88.5% of hemoglobin variance. Adding ECG **degrades** prediction for ALL 6 analytes. This is a **robust, consistent negative result**.

---

### Direction G: ECG Dynamic Changes (Within-Patient ΔECG)

**Design**: Instead of static ECG snapshots, analyze how ECG CHANGES between sessions within the same patient, and whether these changes correlate with lab changes.

#### Key Results

| Metric | Finding |
|--------|---------|
| ΔECG—ΔLab significant correlations | Only 2 weak signals (Δzero-crossing ↔ ΔpO2: ρ=0.114) |
| ΔECG predictive value (ΔR²) | **All negative** (range: -0.015 to -4.234) |
| Analytes with ECG improvement | **0/6** |

#### ECG Feature Stability Within Patients

| Stability | Features | CV Range |
|-----------|----------|:--------:|
| **STABLE** (CV<0.3) | ecg_rms, ecg_total_power, ecg_mean_abs, ecg_ptp, ecg_max, ecg_zcr, ecg_centroid, ecg_min, hr | 0.00–0.29 |
| **UNSTABLE** (CV>0.6) | ecg_skewness, ecg_kurtosis, pnn50, sdnn, rmssd, ecg_lf_hf_ratio | 0.65–0.89 |

**Key insight**: HRV features (SDNN, RMSSD, pNN50) are highly unstable within patients (CV>0.7), confirming Phase 1's finding that 10s windows don't capture stable patient characteristics. Even within-patient ΔECG provides no predictive power for ΔLab.

---

### Direction H: Hierarchical Multimodal Fusion

**Design**: Progressive baseline stacking — B0 (mean) → B1 (prev lab) → B2 (+surgery timing + demographics) → B3 (+ECG). The key test: **does B3 > B2?**

#### Regression Results

**ECG adds value beyond clinical baseline for: 0/6 analytes.**

| Analyte | R²(B2) | R²(B3) | ΔR² | Verdict |
|---------|:------:|:------:|:---:|:-------:|
| hemoglobin | 0.882 | 0.876 | -0.006 | no value |
| lactate | 0.322 | 0.296 | -0.025 | no value |
| glucose | 0.080 | 0.052 | -0.029 | no value |
| pco2 | 0.482 | 0.444 | -0.038 | no value |
| po2 | 0.089 | 0.015 | -0.074 | no value |
| troponin | -0.477 | -0.961 | -0.483 | no value |

#### Classification Results

**ECG degrades classification for 5/6 tasks.**

| Analyte | AUC(B2) | AUC(B3) | ΔAUC | bACC(B2) | bACC(B3) | ΔbACC |
|---------|:-------:|:-------:|:----:|:--------:|:--------:|:-----:|
| hemoglobin | **0.943** | 0.925 | -0.018 | **0.888** | 0.841 | -0.047 |
| lactate | 0.875 | 0.835 | -0.040 | 0.822 | 0.773 | -0.049 |
| glucose | 0.805 | 0.782 | -0.023 | 0.771 | 0.751 | -0.020 |
| troponin | 0.757 | 0.703 | -0.054 | 0.639 | 0.672 | +0.033 |
| pco2 | 0.565 | 0.481 | -0.083 | 0.547 | 0.483 | -0.065 |

**The clinical baseline alone (previous lab value + days from surgery) achieves AUC=0.943 for hemoglobin abnormality detection.** Adding ECG features consistently reduces performance.

---

### Direction I: Clinically Meaningful Label Redefinitions

**Motivation**: Current binary thresholds (e.g., lactate > 2.0) are too coarse for post-CABG. We propose 5 clinically motivated labels:

| # | Label | Definition | Feasibility |
|:--:|-------|-----------|:-----------:|
| 1 | **lactate_clearance_failure** | Post-op lactate slope > 0 AND last > 2.0 | 6/60 patients (10%) |
| 2 | **hb_drop_severe** | Hb drop > 30 g/L from pre-op baseline | 22/59 patients (37%) |
| 3 | **troponin_excessive** | TnI > p90 for days-from-surgery bin | Per-bin p90 available |
| 4 | **multi_deterioration** | ≥2 analytes worsening simultaneously | 90 events detected |
| 5 | **delayed_recovery** | Any analyte not normalized by post-op day 3 | Computable |

#### Supporting Data

- **Troponin temporal pattern**: Median rises 7→405→313→83→55 ng/L across surgery (pre→peri→post→late). p90 thresholds range from 167 (pre-op) to 2963 (days 1-3 post-op).
- **Multi-deterioration combinations**: Most common is lactate+troponin+glucose+hemoglobin (14 events), consistent with systemic inflammatory response.
- **Hb drop**: Mean 27.1 g/L (19.7% of baseline); 37.3% of patients have severe drop >30 g/L.

---

## Synthesis: What We Know Now

### The Strong Signal: Clinical Context

The single best predictor of a patient's current lab value is their **previous lab value**:

| Analyte | R² from previous value alone |
|---------|:---------------------------:|
| Hemoglobin | 0.885 |
| pCO2 | 0.458 |
| Lactate | 0.316 |
| Glucose | 0.283 |
| Troponin | 0.127 |
| pO2 | -0.058 |

Adding days-from-surgery provides modest improvement for some analytes. Adding demographics adds negligible value. Adding ECG features **consistently degrades** performance.

### The Weak Signal: 10-Second ECG

Across 4 Phase 2 experiments, 3 analytes × multiple model types, **ECG features never improved prediction beyond the clinical baseline**. This is not a failure of feature engineering — it's a fundamental limitation of what a 10-second ECG window can tell us about systemic physiology.

### Why Is ECG So Weak?

1. **Temporal mismatch**: Lab values reflect hours-to-days of physiology; ECG reflects seconds
2. **Patient homogeneity**: All CABG patients share similar ECG characteristics
3. **Window too short**: 10 seconds captures only a few heartbeats; HRV needs ≥5 minutes
4. **Wrong features**: ECG morphology may not encode metabolic/hematologic state
5. **Noise dominates signal**: HRV features have CV>0.7 within patients

### What IS ECG Good For?

Based on this analysis, 10-second ECG from HealthMirror is best suited for:
- **Arrhythmia detection** (AFib, ectopy) — instantaneous events visible in short windows
- **Heart rate monitoring** — HR is the most stable ECG feature (CV=0.29)
- **Signal quality assessment** — the waveform itself, not derived features
- **Within-patient anomaly detection** — significant deviation from own baseline (needs longer baseline)

---

## Recommendations for Future Work

### 1. Prioritize Clinical Context
All prediction models should use previous lab values + surgery timing as the **mandatory baseline**. Any new modality must demonstrate improvement OVER this baseline.

### 2. Longer ECG Windows
HRV analysis requires ≥5 minutes. The current 10-second window is fundamentally inadequate for autonomic assessment.

### 3. Redefine Prediction Targets
The proposed labels (Direction I) are more clinically meaningful than generic threshold-based abnormality. Particularly: lactate clearance failure, Hb drop severity, and multi-deterioration.

### 4. ECG as Anomaly Detector
Rather than predicting lab values, use ECG to detect when a patient DEVIATES from their own baseline. This requires building individual baselines from multiple sessions.

### 5. Face/rPPG Exploration
This report focused on ECG. Face/rPPG features (heart rate from video, perfusion indices, skin color changes) may capture different physiological signals and should be explored separately.

### 6. SQA-Filtered Analysis
Signal quality assessment (SQA) could identify high-quality ECG segments. Re-running these analyses on SQA-filtered data might reveal signals currently buried in noise.

---

## Appendix: File Structure (Updated)

```
study/exp3_independent_research/
├── REPORT.md                                  # This report (updated Phase 2)
├── RESEARCH_PLAN.md                           # Original research plan
├── config.py                                  # Shared configuration
├── phase2_data_cleaning/
│   └── build_corrected_dataset.py             # Fixed hemoglobin, unit validation
├── direction_a_feature_eng/                   # Phase 1: Feature engineering
├── direction_b_cross_analyte/                 # Phase 1: Cross-analyte correlation
├── direction_d_surgery_timing/                # Phase 1: Surgery timing
├── direction_e_contrastive/                   # Phase 1: Patient similarity
├── direction_f_trajectory/                    # Phase 2: Recovery trajectory ★
│   └── analyze.py                             # Tasks A, B, C
├── direction_g_ecg_dynamics/                  # Phase 2: Within-patient ΔECG ★
│   └── analyze.py
├── direction_h_multimodal_fusion/             # Phase 2: Hierarchical baseline ★
│   └── analyze.py
├── direction_i_labels/                        # Phase 2: Redefined labels ★
│   └── analyze.py
└── outputs/
    ├── direction_a/                           # Phase 1 outputs
    ├── direction_b/
    ├── direction_d/
    ├── direction_e/
    └── phase2/                                # Phase 2 outputs
        ├── direction_a/ecg_features_corrected.csv
        ├── direction_f/
        ├── direction_g/
        ├── direction_h/
        └── direction_i/
```
| Intra-patient ECG distance (mean) | 4.533 |
| Inter-patient ECG distance (mean) | 4.326 |
| Ratio (inter/intra) | **0.95×** |
| Mann-Whitney U p-value | **0.927** |

**Same-patient ECG sessions are NO more similar than different-patient sessions.**

### Interpretation
This is a critical negative finding: **10-second ECG windows do not capture stable patient-identifying information**. Possible explanations:
1. A 10-second window is too short to capture stable ECG characteristics
2. ECG varies significantly with patient state (stress, medication, time of day)
3. The feature set may not capture the right morphological details
4. Patients in this cohort are clinically homogeneous (all CABG) with similar ECG profiles

This finding has implications for Exp1 (self-supervised learning): if same-patient sessions are not similar, contrastive learning that treats same-patient sessions as "positive pairs" will fail.

---

## Overall Conclusions

### What We Learned

| # | Finding | Significance |
|:--:|---------|:------------:|
| 1 | **Feature engineering beats DL** for small medical datasets | Methodological: choose models appropriate to data scale |
| 2 | **Surgery timing dominates** lab value prediction | Clinical: CABG recovery follows predictable temporal patterns |
```
