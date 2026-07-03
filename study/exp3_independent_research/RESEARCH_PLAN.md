# Exp3: Independent Academic Research — Research Plan

> **Date**: 2026-07-03
> **Goal**: Explore novel scientific questions using the HealthMirror multimodal dataset beyond existing Exp1 (masked reconstruction) and Exp2 (lab abnormality prediction).

---

## 0. Data Inventory

### Available Modalities

| Modality | Description | Dimensions | Source |
|----------|------------|------------|--------|
| ECG | Cleaned 10s ECG signal | (N, 256) | `mirror*_auto_cleaned_sqi/patient_*.csv` |
| Face | Single frame from RGB video | (N, 32, 32) | Extracted from video.avi |
| Lab values | Time-series lab measurements | 132,771 rows | `merged_lab_tests.csv` |
| Patient info | Demographics, BP, HR, SpO2 | Per-session | `cleaned_patient_info.csv` |

### Data Scale
- **2,780** cleaned ECG sessions from **7 mirrors** (1,2,4,5,6,7)
- **349** unique patients with lab data (132,771 measurements)
- **731** samples matched between ECG sessions and lab values (Exp2 manifest)
- **6** core analytes: lactate, troponin, glucose, hemoglobin, pO2, pCO2
- Patient population: **cardiac surgery patients** (predominantly CABG)

### Key Data Characteristics
1. **Temporal**: Lab values measured multiple times per patient (pre/post surgery)
2. **Hierarchical**: Multiple ECG sessions per patient, nested within mirrors
3. **Imbalanced**: Some abnormal conditions are rare (<2% positive)
4. **Time-gapped**: Lab measurements may be hours/days from ECG capture
5. **Clinical context**: Most patients undergo CABG surgery

---

## 1. Research Directions

### Direction A: ECG-Derived Physiological Feature Engineering
**Question**: Can classical ECG features (HR, HRV, morphological features) better predict lab abnormalities than raw DL?

**Rationale**: Exp2 showed raw DL struggles. Classical features are more interpretable and may capture clinically relevant information that DL models lose in the noise.

**Method**:
- Extract HR, HRV (SDNN, RMSSD, pNN50), and morphological features from each ECG window
- Train lightweight classifiers (logistic regression, random forest, XGBoost) on these features
- Compare with Exp2 DL results

**Expected output**: Feature importance rankings, comparison with DL baseline

---

### Direction B: Cross-Analyte Correlation Analysis in Cardiac Surgery Context
**Question**: How do different lab analytes correlate with each other, and can ECG features predict multi-analyte patterns?

**Rationale**: In cardiac surgery, multiple physiological systems are affected simultaneously (e.g., tissue perfusion → lactate↑, pO2↓). Understanding these correlations could reveal systemic patterns.

**Method**:
- Compute correlation matrix of all lab analytes across time-matched samples
- Cluster patients by multi-analyte profiles
- Analyze whether ECG features differ across clusters
- Build models predicting multi-analyte "risk scores"

**Expected output**: Correlation matrices, patient clustering results, cluster-vs-ECG analysis

---

### Direction C: Lactate Temporal Trajectory Modeling
**Question**: Can we model the temporal trajectory of lactate values around surgery, and does ECG morphology relate to trajectory type?

**Rationale**: Lactate is a key marker of tissue perfusion and is frequently measured post-CABG. The shape of the lactate trajectory (rapid vs slow clearance) has prognostic value.

**Method**:
- For patients with ≥3 lactate measurements, fit trajectory models (linear, exponential decay)
- Classify trajectories: "fast normalizer", "slow normalizer", "non-normalizer"
- Analyze ECG features by trajectory type
- Predict trajectory type from first ECG+face session

**Expected output**: Trajectory classification, ECG-vs-trajectory analysis

---

### Direction D: Surgery-Centric Temporal Modeling
**Question**: How do physiological signals change predictably around the time of surgery?

**Rationale**: CABG surgery is a controlled physiological insult. The recovery pattern may be reflected in ECG/face signals.

**Method**:
- For each ECG session, compute days-relative-to-surgery
- Model lab values as a function of: days_from_surgery + ECG_features
- Analyze which ECG features change most around surgery

**Expected output**: Temporal trend plots, importance of surgery timing vs ECG features

---

### Direction E: Patient-Level Representation Learning via Contrastive Pre-training
**Question**: Can we learn a meaningful "health state embedding" from ECG+face data that captures patient similarity?

**Rationale**: Rather than predicting specific lab values, learn a representation that groups similar patients. This can be used for anomaly detection or risk stratification.

**Method**:
- Use SimCLR-style contrastive learning on ECG sessions
- Positive pairs: different sessions from same patient (same health state)
- Negative pairs: sessions from different patients
- Evaluate embedding quality via: same-patient vs different-patient similarity, correlation with lab values

**Expected output**: Embedding quality metrics, t-SNE visualization, correlation with clinical variables

---

## 2. Experiment Priority & Execution Order

| Priority | Direction | Reason | Complexity |
|----------|-----------|--------|------------|
| 1 (P0) | A: ECG Feature Engineering | Most likely to yield positive results; directly addresses Exp2 limitations | Low |
| 2 (P1) | B: Cross-Analyte Correlation | Quick to run, provides valuable insight | Low |
| 3 (P1) | D: Surgery-Centric Modeling | Leverages unique clinical context | Medium |
| 4 (P2) | C: Lactate Trajectory | Needs temporal modeling, moderately complex | Medium |
| 5 (P2) | E: Contrastive Learning | Computationally heavier, more exploratory | High |

## 3. File Structure

```
study/exp3_independent_research/
├── RESEARCH_PLAN.md          # This document
├── __init__.py
├── config.py                 # Shared configuration
├── outputs/                  # All results
│   ├── direction_a/
│   ├── direction_b/
│   ├── direction_c/
│   ├── direction_d/
│   └── direction_e/
├── direction_a_feature_eng/  # Direction A code
│   ├── extract_features.py
│   ├── train_eval.py
│   └── run.py
├── direction_b_cross_analyte/
│   ├── analyze.py
│   └── run.py
├── direction_c_lactate_trajectory/
│   ├── trajectory.py
│   └── run.py
├── direction_d_surgery_timing/
│   ├── analyze.py
│   └── run.py
├── direction_e_contrastive/
│   ├── model.py
│   ├── train.py
│   └── run.py
└── REPORT.md                 # Final report
```

## 4. Success Criteria

- At least 3 directions produce statistically significant findings
- Each direction has clear methodology, results, and interpretation
- Findings are reproducible (fixed seeds, documented parameters)
- Code is organized and documented, following Exp1/Exp2 patterns
