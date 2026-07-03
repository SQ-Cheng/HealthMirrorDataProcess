"""Direction D: Surgery-Centric Temporal Modeling.

Analyzes how physiological signals change around the time of CABG surgery.
Key questions:
  1. How do lab values change relative to days from surgery?
  2. Can days-from-surgery + ECG features predict lab values better than ECG alone?
  3. Which ECG features change most around the surgery period?
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..config import ANALYTES, OUTPUT_DIR, SEED


def _parse_surgery_date(date_str):
    """Parse surgery start date from string like '2026-01-22 08:10' or '2026-01-22 08:10^...'."""
    if pd.isna(date_str) or str(date_str).strip() in ("-", "", "nan", "None"):
        return None
    # Take first date if multiple (separated by ^)
    first = str(date_str).split("^")[0].strip()
    try:
        return pd.to_datetime(first)
    except Exception:
        return None


def run_surgery_timing_analysis(output_dir=None):
    """Run surgery-centric temporal analysis."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_d")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    features_path = os.path.join(OUTPUT_DIR, "direction_a", "ecg_features.csv")
    if not os.path.exists(features_path):
        print("ERROR: ecg_features.csv not found. Run Direction A first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples from feature dataset")

    # ── Load lab data to get surgery dates ──
    lab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                            "merged_lab_tests.csv")
    lab_raw = pd.read_csv(lab_path, dtype=str, keep_default_na=False)
    print(f"Loaded lab data: {len(lab_raw)} rows")

    # Build hospital_id -> surgery_date lookup
    lab_raw["hospital_id"] = lab_raw["首页病案号"].apply(
        lambda x: str(x).strip().rstrip(".0").lstrip("0") if str(x).strip() != "nan" else ""
    )
    lab_raw = lab_raw[lab_raw["hospital_id"] != ""]

    # Get surgery dates
    surgery_dates = {}
    for hid, group in lab_raw.groupby("hospital_id"):
        surgery_str = group["手术开始日期"].iloc[0]
        surgery_dt = _parse_surgery_date(surgery_str)
        if surgery_dt is not None:
            surgery_dates[hid] = surgery_dt

    print(f"Patients with known surgery date: {len(surgery_dates)}")

    # Add days_from_surgery to each ECG session
    df["capture_datetime"] = pd.to_datetime(df["capture_time_unix"], unit="s")
    df["days_from_surgery"] = np.nan
    for idx, row in df.iterrows():
        hid = str(row["hospital_id"])
        if hid in surgery_dates:
            surgery_dt = surgery_dates[hid]
            days = (row["capture_datetime"] - surgery_dt).total_seconds() / 86400.0
            df.at[idx, "days_from_surgery"] = days

    n_with_surgery = df["days_from_surgery"].notna().sum()
    print(f"Samples with surgery date: {n_with_surgery}")

    surgery_df = df[df["days_from_surgery"].notna()].copy()
    print(f"Days from surgery range: {surgery_df['days_from_surgery'].min():.0f} to "
          f"{surgery_df['days_from_surgery'].max():.0f}")

    # ── 1. Lab values vs days from surgery ──
    print("\n" + "=" * 60)
    print("1. Lab Values vs Days from Surgery")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    trend_stats = {}
    for i, analyte in enumerate(ANALYTES):
        ax = axes[i]
        val_col = f"{analyte}_value"
        valid = surgery_df[val_col].dropna()

        if len(valid) < 10:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            ax.set_title(analyte)
            continue

        x = surgery_df.loc[valid.index, "days_from_surgery"]
        y = valid.values

        ax.scatter(x, y, alpha=0.3, s=10, c="#1f77b4")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.5, label="Surgery day")

        # Fit LOWESS-like simple moving average
        x_sorted_idx = np.argsort(x)
        x_sorted = x.iloc[x_sorted_idx]
        y_sorted = y[x_sorted_idx]
        # Rolling mean
        window = max(5, len(x_sorted) // 20)
        if len(x_sorted) > window:
            y_smooth = pd.Series(y_sorted).rolling(window=window, center=True, min_periods=3).mean()
            ax.plot(x_sorted, y_smooth, "r-", linewidth=2, alpha=0.8, label="Trend")

        ax.set_xlabel("Days from Surgery")
        ax.set_ylabel(analyte)
        ax.set_title(f"{analyte} vs Surgery Timing")

        # Correlation
        if len(x) > 5:
            r, p = stats.spearmanr(x, y)
            trend_stats[analyte] = {"spearman_r": float(r), "spearman_p": float(p), "n": len(x)}
            ax.text(0.05, 0.95, f"ρ={r:.3f}, p={p:.4f}", transform=ax.transAxes,
                    fontsize=9, verticalalignment="top")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lab_vs_surgery_timing.png"), dpi=150)
    plt.close()
    print("Saved lab_vs_surgery_timing.png")

    print("\nSpearman correlation: lab value vs days from surgery:")
    for a, s in sorted(trend_stats.items(), key=lambda x: abs(x[1]["spearman_r"]), reverse=True):
        sig = "*" if s["spearman_p"] < 0.05 else ""
        print(f"  {a:15s}: ρ={s['spearman_r']:+.3f}, p={s['spearman_p']:.4f} {sig}  (n={s['n']})")

    # ── 2. Can days_from_surgery + ECG predict lab values better? ──
    print("\n" + "=" * 60)
    print("2. Does Surgery Timing Improve Lab Value Prediction?")
    print("=" * 60)

    ecg_feat_cols = [
        "hr", "sdnn", "rmssd", "pnn50",
        "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
        "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
        "ecg_total_power", "ecg_lf_hf_ratio",
    ]

    comparison_results = []
    for analyte in ANALYTES:
        val_col = f"{analyte}_value"
        valid = surgery_df[[val_col] + ecg_feat_cols + ["days_from_surgery", "hospital_id"]].dropna(subset=[val_col])

        if len(valid) < 20:
            continue

        X_ecg = valid[ecg_feat_cols].fillna(valid[ecg_feat_cols].median())
        X_ecg = X_ecg.replace([np.inf, -np.inf], 0.0)

        X_both = X_ecg.copy()
        X_both["days_from_surgery"] = valid["days_from_surgery"]
        X_both["days_abs"] = np.abs(valid["days_from_surgery"])  # distance from surgery

        y = valid[val_col]
        groups = valid["hospital_id"]

        gkf = GroupKFold(n_splits=5)

        # ECG only
        r2_ecg = []
        mae_ecg = []
        # ECG + surgery timing
        r2_both = []
        mae_both = []

        for train_idx, test_idx in gkf.split(X_ecg, y, groups):
            for X_data, metric_list in [(X_ecg, (r2_ecg, mae_ecg)),
                                         (X_both, (r2_both, mae_both))]:
                X_tr = X_data.iloc[train_idx]
                X_te = X_data.iloc[test_idx]
                y_tr = y.iloc[train_idx]
                y_te = y.iloc[test_idx]

                scaler_local = StandardScaler()
                X_tr_s = scaler_local.fit_transform(X_tr)
                X_te_s = scaler_local.transform(X_te)

                rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                           random_state=SEED, n_jobs=-1)
                rf.fit(X_tr_s, y_tr)
                y_pred = rf.predict(X_te_s)

                metric_list[0].append(r2_score(y_te, y_pred))
                metric_list[1].append(mean_absolute_error(y_te, y_pred))

        r2_ecg_mean = np.mean(r2_ecg)
        r2_both_mean = np.mean(r2_both)
        mae_ecg_mean = np.mean(mae_ecg)
        mae_both_mean = np.mean(mae_both)

        improvement = r2_both_mean - r2_ecg_mean
        comparison_results.append({
            "analyte": analyte,
            "r2_ecg_only": float(r2_ecg_mean),
            "r2_ecg_surgery": float(r2_both_mean),
            "mae_ecg_only": float(mae_ecg_mean),
            "mae_ecg_surgery": float(mae_both_mean),
            "r2_improvement": float(improvement),
            "n_samples": len(valid),
        })

        print(f"\n{analyte} (n={len(valid)}):")
        print(f"  ECG only:      R²={r2_ecg_mean:.4f}, MAE={mae_ecg_mean:.4f}")
        print(f"  ECG + surgery:  R²={r2_both_mean:.4f}, MAE={mae_both_mean:.4f}")
        print(f"  Improvement:    ΔR²={improvement:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary: Surgery Timing Contribution")
    print("=" * 60)

    comp_df = pd.DataFrame(comparison_results)
    comp_df.to_csv(os.path.join(output_dir, "surgery_timing_comparison.csv"), index=False)

    avg_improvement = comp_df["r2_improvement"].mean()
    print(f"Average R² improvement from surgery timing: {avg_improvement:+.4f}")

    # Which analytes benefit most?
    best = comp_df.loc[comp_df["r2_improvement"].idxmax()]
    print(f"Most improved: {best['analyte']} (ΔR²={best['r2_improvement']:+.4f})")

    # ── 3. ECG features that change around surgery ──
    print("\n" + "=" * 60)
    print("3. ECG Feature Changes Around Surgery")
    print("=" * 60)

    # Define periods: pre-surgery (<-1 day), peri-surgery (-1 to 3 days), post-surgery (>3 days)
    surgery_df["period"] = pd.cut(surgery_df["days_from_surgery"],
                                   bins=[-np.inf, -1, 3, np.inf],
                                   labels=["pre", "peri", "post"])

    period_counts = surgery_df["period"].value_counts()
    print(f"Period distribution: {dict(period_counts)}")

    print("\nECG features with significant peri-operative changes (ANOVA):")
    for feat in ecg_feat_cols:
        groups = [surgery_df[surgery_df["period"] == p][feat].dropna().values
                  for p in ["pre", "peri", "post"]]
        groups = [g for g in groups if len(g) > 5]
        if len(groups) >= 2:
            try:
                # Use Kruskal-Wallis (non-parametric)
                h, p = stats.kruskal(*groups)
                if p < 0.05:
                    means = [np.mean(g) for g in groups]
                    print(f"  {feat:30s}: H={h:.2f}, p={p:.4f}, "
                          f"pre={means[0]:.2f}, peri={means[1]:.2f}, post={means[2]:.2f}")
            except Exception:
                pass

    # Save
    surgery_df.to_csv(os.path.join(output_dir, "surgery_annotated_samples.csv"), index=False)
    print(f"\nResults saved to {output_dir}/")

    return {
        "trend_stats": trend_stats,
        "avg_r2_improvement": float(avg_improvement),
        "comparison_df": comp_df,
    }


if __name__ == "__main__":
    run_surgery_timing_analysis()
