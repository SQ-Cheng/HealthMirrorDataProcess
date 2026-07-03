"""Direction G: ECG Dynamic Changes — Within-Patient ΔECG Analysis.

Instead of asking "what does a 10s ECG look like?", ask:
  "How does a patient's ECG CHANGE between sessions, and do these changes
   correlate with lab value changes?"

Key insight: many physiological signals are meaningful as RELATIVE changes
from a patient's own baseline, not as absolute values.

Method:
  1. For each patient with ≥2 ECG sessions, compute ΔECG features between
     consecutive sessions
  2. Compute corresponding Δlab values
  3. Use mixed-effects: Δlab ~ days_from_surgery + ΔECG + (1|patient)
  4. Test if ΔECG explains variance beyond surgery timing
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
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


def run_ecg_dynamics_analysis(output_dir=None):
    """Analyze within-patient ECG changes and their relationship to lab changes."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "phase2", "direction_g")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load corrected dataset ──
    features_path = os.path.join(OUTPUT_DIR, "phase2", "direction_a",
                                  "ecg_features_corrected.csv")
    if not os.path.exists(features_path):
        print("ERROR: Corrected dataset not found.")
        return

    df = pd.read_csv(features_path)
    df = df.sort_values(["hospital_id", "capture_time_unix"])
    print(f"Loaded {len(df)} samples, {df['hospital_id'].nunique()} patients")

    # ── Compute within-patient deltas ──
    print("\n" + "=" * 70)
    print("1. Computing Within-Patient Deltas (ΔECG, ΔLab)")
    print("=" * 70)

    delta_rows = []
    for hid, group in df.groupby("hospital_id"):
        group = group.sort_values("capture_time_unix")
        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            row_prev = group.iloc[i]
            row_curr = group.iloc[i + 1]

            time_diff_h = (row_curr["capture_time_unix"] - row_prev["capture_time_unix"]) / 3600.0

            # Only consider sessions within 72 hours of each other
            if time_diff_h > 72:
                continue

            delta_row = {
                "hospital_id": hid,
                "time_diff_h": time_diff_h,
                "days_from_surgery_curr": row_curr.get("days_from_surgery", np.nan),
                "days_from_surgery_prev": row_prev.get("days_from_surgery", np.nan),
            }

            # ΔECG features
            for feat in ECG_FEAT_COLS:
                if pd.notna(row_curr.get(feat)) and pd.notna(row_prev.get(feat)):
                    delta_row[f"delta_{feat}"] = float(row_curr[feat]) - float(row_prev[feat])
                else:
                    delta_row[f"delta_{feat}"] = np.nan

            # ΔLab values
            for a in ANALYTES:
                val_curr = row_curr.get(f"{a}_value")
                val_prev = row_prev.get(f"{a}_value")
                if pd.notna(val_curr) and pd.notna(val_prev):
                    delta_row[f"delta_{a}"] = float(val_curr) - float(val_prev)
                    delta_row[f"{a}_prev"] = float(val_prev)
                else:
                    delta_row[f"delta_{a}"] = np.nan
                    delta_row[f"{a}_prev"] = np.nan

            delta_rows.append(delta_row)

    delta_df = pd.DataFrame(delta_rows)
    print(f"Within-patient deltas: {len(delta_df)} pairs")

    if len(delta_df) < 10:
        print("Insufficient data for delta analysis.")
        return

    # ── Correlation: ΔECG vs ΔLab ──
    print("\n" + "=" * 70)
    print("2. ΔECG — ΔLab Correlation Analysis")
    print("=" * 70)

    delta_ecg_cols = [f"delta_{f}" for f in ECG_FEAT_COLS]

    # For each analyte, find ECG features whose Δ correlates with Δlab
    sig_correlations = []
    for a in ANALYTES:
        delta_lab = delta_df[f"delta_{a}"].dropna()
        if len(delta_lab) < 20:
            continue

        print(f"\n{a} (n={len(delta_lab)} pairs):")
        for ecg_col in delta_ecg_cols:
            valid = delta_df[[ecg_col, f"delta_{a}"]].dropna()
            if len(valid) < 20:
                continue
            r, p = stats.spearmanr(valid[ecg_col], valid[f"delta_{a}"])
            if p < 0.05 and abs(r) > 0.1:
                ecg_name = ecg_col.replace("delta_", "Δ")
                print(f"  {ecg_name:35s}: ρ={r:+.3f}, p={p:.4f}, n={len(valid)}")
                sig_correlations.append({
                    "analyte": a, "ecg_feature": ecg_col,
                    "spearman_r": float(r), "p_value": float(p), "n": len(valid),
                })

    if not sig_correlations:
        print("\n  No significant ΔECG—ΔLab correlations found "
              "(|ρ| > 0.1, p < 0.05).")

    # ── Predictive Model: Δlab ~ days_from_surgery + ΔECG ──
    print("\n" + "=" * 70)
    print("3. ΔLab Prediction: Surgery Timing vs ΔECG")
    print("=" * 70)

    task_g_results = {}
    for a in ANALYTES:
        delta_col = f"delta_{a}"
        valid = delta_df[[delta_col, "days_from_surgery_curr",
                          "time_diff_h", f"{a}_prev",
                          "hospital_id"] + delta_ecg_cols].dropna(
            subset=[delta_col, "days_from_surgery_curr"])

        if len(valid) < 30:
            continue

        y = valid[delta_col].values
        groups = valid["hospital_id"].values

        # Baseline: days_from_surgery + time_diff + previous value
        X_baseline = np.column_stack([
            valid["days_from_surgery_curr"].values,
            np.abs(valid["days_from_surgery_curr"].values),
            valid["time_diff_h"].values,
        ])
        prev_vals = valid[f"{a}_prev"].values
        if np.isfinite(prev_vals).all():
            X_baseline = np.column_stack([X_baseline, prev_vals])

        # + ΔECG
        X_ecg = valid[delta_ecg_cols].fillna(0.0).values
        X_ecg = np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0)
        X_both = np.hstack([X_baseline, X_ecg])

        gkf = GroupKFold(n_splits=5)
        r2_base = []
        r2_both = []
        mae_base = []
        mae_both = []

        for tr_idx, te_idx in gkf.split(X_baseline, y, groups):
            for X_data, r2_l, mae_l in [
                (X_baseline, r2_base, mae_base),
                (X_both, r2_both, mae_both),
            ]:
                X_tr, X_te = X_data[tr_idx], X_data[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                scl = StandardScaler()
                model = Ridge(alpha=1.0, random_state=SEED)
                model.fit(scl.fit_transform(X_tr), y_tr)
                y_pred = model.predict(scl.transform(X_te))
                r2_l.append(r2_score(y_te, y_pred))
                mae_l.append(mean_absolute_error(y_te, y_pred))

        r2_b = np.mean(r2_base)
        r2_bth = np.mean(r2_both)
        delta_r2 = r2_bth - r2_b

        task_g_results[a] = {
            "n_pairs": len(valid),
            "r2_baseline": float(r2_b),
            "r2_with_ecg": float(r2_bth),
            "delta_r2": float(delta_r2),
            "mae_baseline": float(np.mean(mae_base)),
            "mae_with_ecg": float(np.mean(mae_both)),
        }

        print(f"\n{a} ({len(valid)} pairs):")
        print(f"  Baseline (surgery timing + prev value): R²={r2_b:+.4f}")
        print(f"  + ΔECG features:                        R²={r2_bth:+.4f}")
        print(f"  ΔECG contribution:                      ΔR²={delta_r2:+.4f}")

    # ── Summary ──
    print("\n" + "-" * 70)
    print("Direction G Summary:")
    pos_count = sum(1 for v in task_g_results.values() if v["delta_r2"] > 0)
    print(f"  Analytes where ΔECG improves prediction: {pos_count}/{len(task_g_results)}")
    for a, r in sorted(task_g_results.items(), key=lambda x: x[1]["delta_r2"], reverse=True):
        print(f"  {a:15s}: ΔR²={r['delta_r2']:+.4f} (baseline R²={r['r2_baseline']:+.3f})")

    # ── Visualize: ΔECG stability ──
    print("\n" + "=" * 70)
    print("4. ECG Feature Stability Within Patients")
    print("=" * 70)

    # Coefficient of variation for each ECG feature within patients
    cv_data = []
    for hid, group in df.groupby("hospital_id"):
        if len(group) < 2:
            continue
        for feat in ECG_FEAT_COLS:
            vals = group[feat].dropna()
            if len(vals) < 2:
                continue
            cv = np.std(vals) / max(np.abs(np.mean(vals)), 1e-6)
            cv_data.append({"hospital_id": hid, "feature": feat, "cv": cv})

    cv_df = pd.DataFrame(cv_data)
    cv_summary = cv_df.groupby("feature")["cv"].agg(["mean", "median", "std"]).sort_values("median")

    print("Within-patient coefficient of variation (CV) for ECG features:")
    print(f"{'Feature':35s} {'CV_median':>8s} {'CV_mean':>8s} {'Stability':>12s}")
    for feat, row in cv_summary.iterrows():
        stability = "STABLE" if row["median"] < 0.3 else ("MODERATE" if row["median"] < 0.6 else "UNSTABLE")
        print(f"  {feat:33s} {row['median']:8.3f} {row['mean']:8.3f} {stability:>12s}")

    cv_summary.to_csv(os.path.join(output_dir, "ecg_feature_stability.csv"))

    # ── Save ──
    delta_df.to_csv(os.path.join(output_dir, "within_patient_deltas.csv"), index=False)
    pd.DataFrame(task_g_results).T.to_csv(os.path.join(output_dir, "delta_prediction_results.csv"))
    print(f"\nResults saved to {output_dir}/")

    return delta_df, task_g_results, cv_summary


if __name__ == "__main__":
    run_ecg_dynamics_analysis()
