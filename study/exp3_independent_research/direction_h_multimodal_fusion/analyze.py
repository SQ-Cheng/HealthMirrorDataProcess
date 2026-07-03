"""Direction H: Hierarchical Multimodal Fusion with Clinical Backbone.

Design: progressive baselines showing incremental value of each modality.
  B0: mean / constant predictor
  B1: previous lab value (strongest single predictor)
  B2: B1 + days_from_surgery + demographics (clinical backbone)
  B3: B2 + ECG features (physiological signals)
  → Key question: does B3 > B2? (i.e., ECG adds value beyond clinical context)

Note: Face/rPPG features are NOT available in the feature dataset.
      This analysis focuses on the clinical-baseline + ECG paradigm.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from ..config import OUTPUT_DIR, SEED

ANALYTES = ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]

ECG_FEAT_COLS = [
    "hr", "sdnn", "rmssd", "pnn50",
    "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
    "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
    "ecg_total_power", "ecg_lf_hf_ratio", "ecg_spectral_centroid",
]


def run_multimodal_fusion_analysis(output_dir=None):
    """Run hierarchical multimodal fusion analysis."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "phase2", "direction_h")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    features_path = os.path.join(OUTPUT_DIR, "phase2", "direction_a",
                                  "ecg_features_corrected.csv")
    if not os.path.exists(features_path):
        print("ERROR: Corrected dataset not found.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples, {df['hospital_id'].nunique()} patients")

    # ═══════════════════════════════════════════════════════════════════
    # Part 1: Regression — predict continuous lab values
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: Hierarchical Regression — Predicting Lab Values")
    print("=" * 70)
    print("B0: mean  |  B1: prev_lab  |  B2: +days+demographics  |  B3: +ECG")
    print("-" * 70)

    reg_results = {}

    for analyte in ANALYTES:
        val_col = f"{analyte}_value"
        prev_col = f"{analyte}_prev_value"

        # Need: current value, previous value, surgery timing
        cols_needed = [val_col, prev_col, "days_from_surgery", "age", "sex",
                       "hospital_id"] + ECG_FEAT_COLS
        valid = df[cols_needed].dropna(subset=[val_col, "days_from_surgery"])

        # For B1+, need previous value
        has_prev = valid[prev_col].notna()
        if has_prev.sum() < 30:
            print(f"\n{analyte}: insufficient data (n={has_prev.sum()})")
            continue

        valid_b = valid[has_prev].copy()
        y = valid_b[val_col].values
        groups = valid_b["hospital_id"].values
        n = len(y)

        # B0: mean of training set
        # B1: previous value
        X_b1 = valid_b[[prev_col]].values
        # B2: previous + days + demographics
        X_b2 = np.column_stack([
            valid_b[prev_col].values,
            valid_b["days_from_surgery"].values,
            np.abs(valid_b["days_from_surgery"].values),
        ])
        # Add sex as binary
        sex_binary = (valid_b["sex"] == "男").astype(float).values
        X_b2 = np.column_stack([X_b2, sex_binary])
        # Add age if available
        age_vals = valid_b["age"].fillna(valid_b["age"].median()).values
        X_b2 = np.column_stack([X_b2, age_vals])

        # B3: B2 + ECG
        X_ecg = valid_b[ECG_FEAT_COLS].fillna(valid_b[ECG_FEAT_COLS].median()).values
        X_ecg = np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0)
        X_b3 = np.hstack([X_b2, X_ecg])

        # 5-fold CV
        gkf = GroupKFold(n_splits=5)
        scores = {"B0": [], "B1": [], "B2": [], "B3": []}
        maes = {"B0": [], "B1": [], "B2": [], "B3": []}

        for tr_idx, te_idx in gkf.split(X_b2, y, groups):
            y_tr, y_te = y[tr_idx], y[te_idx]

            # B0
            mean_tr = np.mean(y_tr)
            scores["B0"].append(r2_score(y_te, np.full_like(y_te, mean_tr)))
            maes["B0"].append(mean_absolute_error(y_te, np.full_like(y_te, mean_tr)))

            for name, X_data in [("B1", X_b1), ("B2", X_b2), ("B3", X_b3)]:
                X_tr, X_te = X_data[tr_idx], X_data[te_idx]
                scl = StandardScaler()
                m = Ridge(alpha=1.0, random_state=SEED)
                m.fit(scl.fit_transform(X_tr), y_tr)
                p = m.predict(scl.transform(X_te))
                scores[name].append(r2_score(y_te, p))
                maes[name].append(mean_absolute_error(y_te, p))

        # Bootstrap test B3 vs B2
        n_boot = 1000
        better_count = 0
        rng = np.random.default_rng(SEED)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                gkf_b = GroupKFold(n_splits=3)
                for tr_i, te_i in gkf_b.split(X_b2[idx], y[idx], groups[idx]):
                    scl2 = StandardScaler()
                    m2 = Ridge(alpha=1.0).fit(scl2.fit_transform(X_b2[idx][tr_i]), y[idx][tr_i])
                    p2 = m2.predict(scl2.transform(X_b2[idx][te_i]))
                    m3 = Ridge(alpha=1.0).fit(scl2.fit_transform(X_b3[idx][tr_i]), y[idx][tr_i])
                    p3 = m3.predict(scl2.transform(X_b3[idx][te_i]))
                    if r2_score(y[idx][te_i], p3) > r2_score(y[idx][te_i], p2):
                        better_count += 1
                    break
            except Exception:
                pass

        p_val = 1.0 - better_count / max(n_boot, 1)

        reg_results[analyte] = {
            "n": n,
            "r2_B0": float(np.mean(scores["B0"])),
            "r2_B1": float(np.mean(scores["B1"])),
            "r2_B2": float(np.mean(scores["B2"])),
            "r2_B3": float(np.mean(scores["B3"])),
            "mae_B2": float(np.mean(maes["B2"])),
            "mae_B3": float(np.mean(maes["B3"])),
            "delta_r2_B3_B2": float(np.mean(scores["B3"]) - np.mean(scores["B2"])),
            "p_B3_vs_B2": float(p_val),
            "ecg_adds_value": float(np.mean(scores["B3"])) > float(np.mean(scores["B2"])) and p_val < 0.1,
        }

        print(f"\n{analyte} (n={n}):")
        r2s = {k: f"{np.mean(v):+.4f}" for k, v in scores.items()}
        print(f"  B0 (mean):       R²={r2s['B0']}")
        print(f"  B1 (prev lab):   R²={r2s['B1']}")
        print(f"  B2 (+ clinical): R²={r2s['B2']}")
        print(f"  B3 (+ ECG):      R²={r2s['B3']}")
        delta = np.mean(scores["B3"]) - np.mean(scores["B2"])
        icon = "✓" if delta > 0 and p_val < 0.1 else "✗"
        print(f"  ECG contribution: ΔR²={delta:+.4f} (p={p_val:.3f}) {icon}")

    # Summary
    print("\n" + "=" * 70)
    print("REGRESSION SUMMARY: ECG Incremental Value")
    print("=" * 70)
    print(f"{'Analyte':15s} {'R²(B2)':>8s} {'R²(B3)':>8s} {'ΔR²':>8s} {'p':>8s} {'Verdict':>12s}")
    print("-" * 60)
    for a, r in sorted(reg_results.items(), key=lambda x: x[1]["delta_r2_B3_B2"], reverse=True):
        verdict = "ECG HELPS" if r["ecg_adds_value"] else "no value"
        print(f"  {a:13s} {r['r2_B2']:8.4f} {r['r2_B3']:8.4f} "
              f"{r['delta_r2_B3_B2']:+8.4f} {r['p_B3_vs_B2']:8.3f} {verdict:>12s}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: Classification — predict abnormality labels
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: Hierarchical Classification — Abnormality Prediction")
    print("=" * 70)

    label_map = {
        "lactate": "lactate_abnormal",
        "troponin": "troponin_abnormal",
        "glucose": "glucose_abnormal",
        "hemoglobin": "hemoglobin_low",
        "po2": "po2_abnormal",
        "pco2": "pco2_abnormal",
    }

    cls_results = {}
    for analyte, label_col in label_map.items():
        prev_col = f"{analyte}_prev_value"

        cols_needed = [label_col, prev_col, "days_from_surgery", "hospital_id"] + ECG_FEAT_COLS
        valid = df[cols_needed].dropna(subset=[label_col, "days_from_surgery"])

        # Need: binary label + previous value
        has_prev = valid[prev_col].notna()
        if has_prev.sum() < 30:
            continue

        valid_b = valid[has_prev].copy()
        y = valid_b[label_col].values.astype(int)
        groups = valid_b["hospital_id"].values
        n = len(y)
        n_pos = int(y.sum())

        if n_pos < 5 or (n - n_pos) < 5:
            continue

        # Features
        X_prev = valid_b[[prev_col]].values
        X_b2 = np.column_stack([
            valid_b[prev_col].values,
            valid_b["days_from_surgery"].values,
            np.abs(valid_b["days_from_surgery"].values),
        ])
        X_ecg = valid_b[ECG_FEAT_COLS].fillna(valid_b[ECG_FEAT_COLS].median()).values
        X_ecg = np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0)
        X_b3 = np.hstack([X_b2, X_ecg])

        gkf = GroupKFold(n_splits=5)
        auc_b2, auc_b3 = [], []
        bacc_b2, bacc_b3 = [], []

        for tr_idx, te_idx in gkf.split(X_b2, y, groups):
            for X_data, auc_l, bacc_l in [
                (X_b2, auc_b2, bacc_b2),
                (X_b3, auc_b3, bacc_b3),
            ]:
                X_tr, X_te = X_data[tr_idx], X_data[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                scl = StandardScaler()
                X_tr_s = scl.fit_transform(X_tr)
                X_te_s = scl.transform(X_te)

                # Logistic regression (via Ridge classifier)
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                                         max_iter=1000, random_state=SEED)
                clf.fit(X_tr_s, y_tr)
                y_prob = clf.predict_proba(X_te_s)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                auc_l.append(roc_auc_score(y_te, y_prob))
                bacc_l.append(balanced_accuracy_score(y_te, y_pred))

        cls_results[analyte] = {
            "n": n,
            "n_pos": n_pos,
            "auc_B2": float(np.mean(auc_b2)),
            "auc_B3": float(np.mean(auc_b3)),
            "bacc_B2": float(np.mean(bacc_b2)),
            "bacc_B3": float(np.mean(bacc_b3)),
            "delta_auc": float(np.mean(auc_b3) - np.mean(auc_b2)),
            "delta_bacc": float(np.mean(bacc_b3) - np.mean(bacc_b2)),
        }

        print(f"\n{analyte} (n={n}, pos={n_pos}, rate={n_pos/n:.1%}):")
        print(f"  B2 (prev+days): AUC={np.mean(auc_b2):.3f}, bACC={np.mean(bacc_b2):.3f}")
        print(f"  B3 (+ECG):       AUC={np.mean(auc_b3):.3f}, bACC={np.mean(bacc_b3):.3f}")
        print(f"  ΔAUC={np.mean(auc_b3)-np.mean(auc_b2):+.3f}, "
              f"ΔbACC={np.mean(bacc_b3)-np.mean(bacc_b2):+.3f}")

    # ── Summary tables ──
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY: ECG Incremental Value")
    print("=" * 70)
    print(f"{'Analyte':15s} {'AUC(B2)':>8s} {'AUC(B3)':>8s} {'ΔAUC':>8s} "
          f"{'bACC(B2)':>9s} {'bACC(B3)':>9s} {'ΔbACC':>8s}")
    print("-" * 65)
    for a, r in sorted(cls_results.items(), key=lambda x: x[1]["delta_auc"], reverse=True):
        print(f"  {a:13s} {r['auc_B2']:8.3f} {r['auc_B3']:8.3f} "
              f"{r['delta_auc']:+8.3f} {r['bacc_B2']:9.3f} {r['bacc_B3']:9.3f} "
              f"{r['delta_bacc']:+8.3f}")

    # ── Save ──
    pd.DataFrame(reg_results).T.to_csv(os.path.join(output_dir, "regression_results.csv"))
    pd.DataFrame(cls_results).T.to_csv(os.path.join(output_dir, "classification_results.csv"))
    print(f"\nResults saved to {output_dir}/")

    return reg_results, cls_results


if __name__ == "__main__":
    run_multimodal_fusion_analysis()
