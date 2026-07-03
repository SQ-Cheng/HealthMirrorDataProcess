"""Direction F: Peri-Operative Recovery Trajectory Modeling.

Core idea: Model each patient as a recovery trajectory, not independent snapshots.

Tasks:
  A) Predict Δlab (relative change) from surgery timing + ECG features
  B) Identify abnormal recovery trajectories (deviation from population norm)
  C) Surgery-time baseline → test if ECG provides residual information

Design principle:
  - Use days_from_surgery as the primary predictor (established clinical knowledge)
  - Build a strong clinical baseline first
  - Then test: does ECG add residual predictive power?
  - Report negative findings honestly — they're scientifically valuable
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from ..config import OUTPUT_DIR, SEED

ANALYTES = ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]

# ECG feature columns used throughout
ECG_FEAT_COLS = [
    "hr", "sdnn", "rmssd", "pnn50",
    "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
    "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
    "ecg_total_power", "ecg_lf_hf_ratio", "ecg_spectral_centroid",
]


def run_trajectory_analysis(output_dir=None):
    """Run the full peri-operative trajectory analysis."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "phase2", "direction_f")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load corrected dataset ──
    features_path = os.path.join(OUTPUT_DIR, "phase2", "direction_a",
                                  "ecg_features_corrected.csv")
    if not os.path.exists(features_path):
        # Try Phase 1 path as fallback
        features_path = os.path.join(OUTPUT_DIR, "direction_a", "ecg_features_corrected.csv")
    if not os.path.exists(features_path):
        print("ERROR: Corrected dataset not found. Run phase2_data_cleaning first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples from corrected dataset")

    # Keep only samples with surgery timing
    df_surg = df[df["days_from_surgery"].notna()].copy()
    print(f"Samples with surgery timing: {len(df_surg)}")

    # ═══════════════════════════════════════════════════════════════════
    # TASK A: Predict Δlab (relative change) from surgery + ECG
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK A: Predicting ΔLab (change from previous measurement)")
    print("=" * 70)

    task_a_results = {}
    for analyte in ANALYTES:
        val_col = f"{analyte}_value"
        delta_col = f"{analyte}_delta_value"
        prev_col = f"{analyte}_prev_value"

        # Need both current value and previous value
        valid = df_surg[[val_col, delta_col, prev_col, "days_from_surgery",
                         "hospital_id"] + ECG_FEAT_COLS].dropna(
            subset=[delta_col, "days_from_surgery"])
        if len(valid) < 30:
            print(f"\n{analyte}: insufficient data (n={len(valid)}), skipping")
            continue

        # Also remove rows where delta_col is computed from distant measurements
        # (keep only where previous measurement is within 48 hours)
        delta_h_col = f"{analyte}_delta_h"
        if delta_h_col in df_surg.columns:
            valid_dh = df_surg.loc[valid.index, delta_h_col]
            recent_mask = valid_dh.abs() < 48
            valid = valid[recent_mask.values]

        if len(valid) < 20:
            print(f"\n{analyte}: insufficient recent data, skipping")
            continue

        y = valid[delta_col].values  # Δlab
        groups = valid["hospital_id"].values

        # Feature sets
        X_surgery = valid[["days_from_surgery"]].copy()
        X_surgery["days_abs"] = np.abs(valid["days_from_surgery"])
        X_surgery["prev_value"] = valid[prev_col].values

        X_ecg = valid[ECG_FEAT_COLS].fillna(valid[ECG_FEAT_COLS].median())
        X_ecg = X_ecg.replace([np.inf, -np.inf], 0.0)

        X_both = np.hstack([X_surgery.values, X_ecg.values])
        X_both_cols = list(X_surgery.columns) + ECG_FEAT_COLS

        # 5-fold CV (patient-level)
        gkf = GroupKFold(n_splits=5)
        r2_surgery = []
        mae_surgery = []
        r2_both = []
        mae_both = []

        for tr_idx, te_idx in gkf.split(X_surgery, y, groups):
            for X_data, r2_list, mae_list in [
                (X_surgery.values, r2_surgery, mae_surgery),
                (X_both, r2_both, mae_both),
            ]:
                X_tr = X_data[tr_idx]
                X_te = X_data[te_idx]
                y_tr = y[tr_idx]
                y_te = y[te_idx]

                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)

                # Ridge regression (linear, regularized — appropriate for small data)
                model = Ridge(alpha=1.0, random_state=SEED)
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
                r2_list.append(r2_score(y_te, y_pred))
                mae_list.append(mean_absolute_error(y_te, y_pred))

        r2_s_mean = np.mean(r2_surgery)
        r2_b_mean = np.mean(r2_both)
        mae_s_mean = np.mean(mae_surgery)
        mae_b_mean = np.mean(mae_both)
        delta_r2 = r2_b_mean - r2_s_mean

        # Bootstrap test for significance of ΔR²
        n_bootstrap = 1000
        diffs = []
        rng = np.random.default_rng(SEED)
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y), size=len(y), replace=True)
            y_boot = y[idx]
            g_boot = groups[idx]
            Xs_boot = X_surgery.values[idx]
            Xb_boot = X_both[idx]

            gkf_boot = GroupKFold(n_splits=3)
            try:
                for tr_i, te_i in gkf_boot.split(Xs_boot, y_boot, g_boot):
                    scl = StandardScaler()
                    r_s = Ridge(alpha=1.0).fit(scl.fit_transform(Xs_boot[tr_i]), y_boot[tr_i])
                    p_s = r_s.predict(scl.transform(Xs_boot[te_i]))
                    r_b = Ridge(alpha=1.0).fit(scl.fit_transform(Xb_boot[tr_i]), y_boot[tr_i])
                    p_b = r_b.predict(scl.transform(Xb_boot[te_i]))
                    diffs.append(r2_score(y_boot[te_i], p_b) - r2_score(y_boot[te_i], p_s))
                    break
            except Exception:
                pass

        p_value = np.mean(np.array(diffs) <= 0) if diffs else 1.0

        print(f"\n{analyte} (n={len(valid)}):")
        print(f"  Surgery-only baseline: R²={r2_s_mean:+.4f}, MAE={mae_s_mean:.4f}")
        print(f"  Surgery + ECG:          R²={r2_b_mean:+.4f}, MAE={mae_b_mean:.4f}")
        print(f"  ECG residual value:     ΔR²={delta_r2:+.4f} (p={p_value:.3f})")
        if delta_r2 > 0 and p_value < 0.1:
            print(f"  → ECG provides marginal additional information (p<0.1)")
        else:
            print(f"  → ECG does NOT significantly improve Δlab prediction")

        task_a_results[analyte] = {
            "n": len(valid),
            "r2_surgery": float(r2_s_mean),
            "r2_surgery_ecg": float(r2_b_mean),
            "mae_surgery": float(mae_s_mean),
            "mae_surgery_ecg": float(mae_b_mean),
            "delta_r2": float(delta_r2),
            "p_value": float(p_value),
        }

    # ── TASK A Summary ──
    print("\n" + "-" * 70)
    print("Task A Summary:")
    sig_count = sum(1 for v in task_a_results.values()
                    if v["delta_r2"] > 0 and v["p_value"] < 0.1)
    print(f"  Analytes where ECG adds significant residual: {sig_count}/{len(task_a_results)}")
    for a, r in sorted(task_a_results.items(), key=lambda x: x[1]["delta_r2"], reverse=True):
        sig = " *" if r["delta_r2"] > 0 and r["p_value"] < 0.1 else ""
        print(f"  {a:15s}: ΔR²={r['delta_r2']:+.4f} (p={r['p_value']:.3f}){sig}")

    # ═══════════════════════════════════════════════════════════════════
    # TASK B: Identify Abnormal Recovery Trajectories
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK B: Abnormal Recovery Trajectory Detection")
    print("=" * 70)

    # Fit population recovery curves for each analyte
    # For each analyte: value ~ f(days_from_surgery)
    # Then flag patients whose recovery deviates from population norm

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    trajectory_summary = {}
    for i, analyte in enumerate(ANALYTES):
        ax = axes[i]
        val_col = f"{analyte}_value"
        valid = df_surg[[val_col, "days_from_surgery", "hospital_id"]].dropna(
            subset=[val_col, "days_from_surgery"])

        if len(valid) < 20:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(analyte)
            continue

        x = valid["days_from_surgery"].values
        y = valid[val_col].values

        # Fit a piecewise model: pre-surgery plateau + post-surgery exponential decay/recovery
        # Simplified: fit LOWESS-like rolling median
        x_sorted_idx = np.argsort(x)
        x_sorted = x[x_sorted_idx]
        y_sorted = y[x_sorted_idx]

        # Rolling median for population trend
        window = max(7, len(x_sorted) // 15)
        y_trend = pd.Series(y_sorted).rolling(window=window, center=True, min_periods=5).median()

        # Residual = actual - trend
        residuals = y - np.interp(x, x_sorted, y_trend.fillna(method="bfill").fillna(method="ffill"))

        ax.scatter(x, y, alpha=0.25, s=8, c="#1f77b4", label="Observations")
        ax.plot(x_sorted, y_trend, "r-", linewidth=2, alpha=0.9, label="Population trend")
        ax.axvline(x=0, color="green", linestyle="--", alpha=0.5, label="Surgery day")
        ax.set_xlabel("Days from surgery")
        ax.set_ylabel(analyte)
        ax.set_title(f"{analyte} recovery trajectory")

        # Flag patients with extreme residuals (top/bottom 10%)
        abs_residuals = np.abs(residuals)
        threshold_90 = np.percentile(abs_residuals[~np.isnan(abs_residuals)], 90)
        n_flagged = int(np.sum(abs_residuals > threshold_90))

        trajectory_summary[analyte] = {
            "n_samples": len(valid),
            "residual_std": float(np.std(residuals[~np.isnan(residuals)])),
            "n_flagged": n_flagged,
            "flag_rate": n_flagged / max(len(valid), 1),
        }

        ax.text(0.05, 0.95, f"Flagged: {n_flagged} ({n_flagged/max(len(valid),1):.1%})",
                transform=ax.transAxes, fontsize=9, verticalalignment="top")

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "recovery_trajectories.png"), dpi=150)
    plt.close()
    print("Saved recovery_trajectories.png")

    print("\nTrajectory Summary:")
    for a, s in trajectory_summary.items():
        print(f"  {a:15s}: σ_residual={s['residual_std']:.2f}, "
              f"flagged={s['n_flagged']} ({s['flag_rate']:.1%})")

    # ═══════════════════════════════════════════════════════════════════
    # TASK C: Clinical Baseline vs ECG Residual — STRONG negative control
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TASK C: Hierarchical Baseline — Does ECG Explain Residual?")
    print("=" * 70)
    print("Building progressive baselines:")
    print("  B0: mean predictor (intercept only)")
    print("  B1: days_from_surgery only")
    print("  B2: days_from_surgery + previous_lab_value (strong clinical baseline)")
    print("  B3: B2 + ECG features")
    print("  → Does B3 > B2?")

    task_c_results = {}
    for analyte in ANALYTES:
        val_col = f"{analyte}_value"
        prev_col = f"{analyte}_prev_value"

        valid = df_surg[[val_col, prev_col, "days_from_surgery",
                         "hospital_id"] + ECG_FEAT_COLS].dropna(
            subset=[val_col, "days_from_surgery"])

        if len(valid) < 50:
            continue

        y = valid[val_col].values
        groups = valid["hospital_id"].values

        # B0: mean
        mean_pred = np.full_like(y, np.mean(y))

        # B1: days_from_surgery
        X_b1 = valid[["days_from_surgery"]].values

        # B2: days + previous value
        has_prev = valid[prev_col].notna()
        if has_prev.sum() < 30:
            continue
        X_b2_valid = valid[has_prev]
        y_b2 = y[has_prev]
        groups_b2 = groups[has_prev]
        X_b2 = np.column_stack([
            X_b2_valid["days_from_surgery"].values,
            X_b2_valid[prev_col].values,
        ])
        days_col_b2 = X_b2_valid["days_from_surgery"].values

        # B3: B2 + ECG
        X_ecg_valid = X_b2_valid[ECG_FEAT_COLS].fillna(
            X_b2_valid[ECG_FEAT_COLS].median()).values
        X_ecg_valid = np.nan_to_num(X_ecg_valid, nan=0.0, posinf=0.0, neginf=0.0)
        X_b3 = np.hstack([X_b2, X_ecg_valid])

        # CV evaluation
        gkf = GroupKFold(n_splits=5)
        r2_scores = {"B0": [], "B1": [], "B2": [], "B3": []}
        mae_scores = {"B0": [], "B1": [], "B2": [], "B3": []}

        fold_idx = 0
        for tr_idx, te_idx in gkf.split(X_b2, y_b2, groups_b2):
            fold_idx += 1
            # B0: mean of training set
            y_tr = y_b2[tr_idx]
            y_te = y_b2[te_idx]
            mean_tr = np.mean(y_tr)
            r2_scores["B0"].append(r2_score(y_te, np.full_like(y_te, mean_tr)))
            mae_scores["B0"].append(mean_absolute_error(y_te, np.full_like(y_te, mean_tr)))

            # B1
            X_tr = X_b1[has_prev][tr_idx].reshape(-1, 1)
            X_te = X_b1[has_prev][te_idx].reshape(-1, 1)
            scl = StandardScaler()
            m = Ridge(alpha=1.0).fit(scl.fit_transform(X_tr), y_tr)
            p = m.predict(scl.transform(X_te))
            r2_scores["B1"].append(r2_score(y_te, p))
            mae_scores["B1"].append(mean_absolute_error(y_te, p))

            # B2
            X_tr = X_b2[tr_idx]
            X_te = X_b2[te_idx]
            scl = StandardScaler()
            m = Ridge(alpha=1.0).fit(scl.fit_transform(X_tr), y_tr)
            p = m.predict(scl.transform(X_te))
            r2_scores["B2"].append(r2_score(y_te, p))
            mae_scores["B2"].append(mean_absolute_error(y_te, p))

            # B3
            X_tr = X_b3[tr_idx]
            X_te = X_b3[te_idx]
            scl = StandardScaler()
            m = Ridge(alpha=1.0).fit(scl.fit_transform(X_tr), y_tr)
            p = m.predict(scl.transform(X_te))
            r2_scores["B3"].append(r2_score(y_te, p))
            mae_scores["B3"].append(mean_absolute_error(y_te, p))

        # Bootstrap test B3 vs B2
        n_boot = 1000
        b3_better_count = 0
        rng = np.random.default_rng(SEED)
        for _ in range(n_boot):
            idx = rng.choice(len(y_b2), size=len(y_b2), replace=True)
            try:
                gkf_b = GroupKFold(n_splits=3)
                for tr_i, te_i in gkf_b.split(X_b2[idx], y_b2[idx], groups_b2[idx]):
                    scl2 = StandardScaler()
                    r2_b2 = Ridge(alpha=1.0).fit(scl2.fit_transform(X_b2[idx][tr_i]),
                                                  y_b2[idx][tr_i])
                    p2 = r2_b2.predict(scl2.transform(X_b2[idx][te_i]))
                    r2_b3 = Ridge(alpha=1.0).fit(scl2.fit_transform(X_b3[idx][tr_i]),
                                                  y_b2[idx][tr_i])
                    p3 = r2_b3.predict(scl2.transform(X_b3[idx][te_i]))
                    if r2_score(y_b2[idx][te_i], p3) > r2_score(y_b2[idx][te_i], p2):
                        b3_better_count += 1
                    break
            except Exception:
                pass
        p_b3_vs_b2 = 1.0 - b3_better_count / max(n_boot, 1)

        task_c_results[analyte] = {
            "n": len(y_b2),
            "r2_B0": float(np.mean(r2_scores["B0"])),
            "r2_B1": float(np.mean(r2_scores["B1"])),
            "r2_B2": float(np.mean(r2_scores["B2"])),
            "r2_B3": float(np.mean(r2_scores["B3"])),
            "mae_B2": float(np.mean(mae_scores["B2"])),
            "mae_B3": float(np.mean(mae_scores["B3"])),
            "delta_r2_B3_B2": float(np.mean(r2_scores["B3"]) - np.mean(r2_scores["B2"])),
            "p_b3_vs_b2": float(p_b3_vs_b2),
        }

        print(f"\n{analyte} (n={len(y_b2)}):")
        print(f"  B0 (mean):           R²={np.mean(r2_scores['B0']):+.4f}")
        print(f"  B1 (days_from_surg):  R²={np.mean(r2_scores['B1']):+.4f}")
        print(f"  B2 (+ previous lab):  R²={np.mean(r2_scores['B2']):+.4f}")
        print(f"  B3 (+ ECG features):  R²={np.mean(r2_scores['B3']):+.4f}")
        delta = np.mean(r2_scores["B3"]) - np.mean(r2_scores["B2"])
        if delta > 0 and p_b3_vs_b2 < 0.1:
            print(f"  → ECG adds significant residual info: ΔR²={delta:+.4f} (p={p_b3_vs_b2:.3f}) *")
        else:
            print(f"  → ECG does NOT add significant value beyond clinical baseline (p={p_b3_vs_b2:.3f})")

    # ── Save results ──
    pd.DataFrame(task_a_results).T.to_csv(os.path.join(output_dir, "task_a_delta_lab.csv"))
    pd.DataFrame(task_c_results).T.to_csv(os.path.join(output_dir, "task_c_hierarchical.csv"))

    print(f"\nResults saved to {output_dir}/")

    return task_a_results, trajectory_summary, task_c_results


if __name__ == "__main__":
    run_trajectory_analysis()
