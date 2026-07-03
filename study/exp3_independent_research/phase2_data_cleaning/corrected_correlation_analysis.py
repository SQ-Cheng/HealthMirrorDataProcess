"""Re-analysis: Corrected cross-analyte correlations + partial correlation by days-from-surgery.

Addresses the reviewer's concern that:
  1. Old hemoglobin-pO2 r=-0.553 was contaminated by the g/dL→g/L unit bug
  2. "Anemia → lower oxygen" interpretation was backwards (more Hb should → higher O2 capacity)
  3. Surgery timing is a major confounder — both Hb and pO2 change with days-from-surgery
  4. Need partial correlation to isolate the true physiological coupling

Outputs:
  - Corrected Pearson + Spearman matrices
  - Partial correlation controlling for days-from-surgery
  - Stratified correlation by surgery phase (pre, peri, early-post, late-post)
  - Scatter plots with regression lines
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

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "outputs", "phase2", "direction_b_corrected")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ANALYTES = ["lactate", "troponin", "glucose", "hemoglobin", "po2", "pco2"]

SURGERY_BINS = [
    ("Pre-op (< -7d)", -np.inf, -7),
    ("Pre-op (-7 to -1d)", -7, -1),
    ("Peri-op (-1 to +1d)", -1, 1),
    ("Early post-op (1 to 3d)", 1, 3),
    ("Mid post-op (3 to 7d)", 3, 7),
    ("Late post-op (> 7d)", 7, np.inf),
]


def partial_corr(x, y, z):
    """Compute Pearson partial correlation: r_{xy.z} = (r_{xy} - r_{xz}*r_{yz}) / sqrt((1-r_{xz}^2)*(1-r_{yz}^2))."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 10:
        return np.nan, np.nan
    r_xy = stats.pearsonr(x, y)[0]
    r_xz = stats.pearsonr(x, z)[0]
    r_yz = stats.pearsonr(y, z)[0]
    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if abs(denom) < 1e-10:
        return np.nan, np.nan
    r_xy_z = (r_xy - r_xz * r_yz) / denom
    # P-value via Fisher z-transform
    n = len(x)
    z_val = 0.5 * np.log((1 + r_xy_z) / max(1 - r_xy_z, 1e-10)) * np.sqrt(n - 3)
    from scipy.stats import norm
    p_val = 2 * (1 - norm.cdf(abs(z_val)))
    return r_xy_z, p_val


def main():
    # ── Load corrected data ──
    features_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "phase2", "direction_a", "ecg_features_corrected.csv")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")

    # ═══════════════════════════════════════════════════════════════════
    # 1. CORRECTED PEARSON + SPEARMAN MATRICES
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("1. CORRECTED PEARSON CORRELATION MATRIX")
    print("=" * 70)

    value_cols = [f"{a}_value" for a in ANALYTES]
    corr_df = df[value_cols].copy()
    corr_df.columns = ANALYTES

    # Pearson
    pearson = corr_df.corr(method="pearson")
    # Spearman
    spearman = corr_df.corr(method="spearman")

    print("\nPearson (corrected):")
    print(pearson.to_string(float_format="+.4f"))

    print("\nSpearman (corrected):")
    print(spearman.to_string(float_format="+.4f"))

    # ── P-values for key pairs ──
    key_pairs = [
        ("hemoglobin", "po2"),
        ("hemoglobin", "glucose"),
        ("hemoglobin", "lactate"),
        ("glucose", "po2"),
        ("glucose", "lactate"),
        ("glucose", "troponin"),
        ("lactate", "po2"),
        ("lactate", "troponin"),
        ("hemoglobin", "troponin"),
        ("glucose", "hemoglobin"),
        ("lactate", "hemoglobin"),
        ("po2", "lactate"),
    ]

    print(f"\n{'Pair':>25s}  {'Pearson r':>9s}  {'p':>8s}  {'Spearman ρ':>10s}  {'p':>8s}  {'n':>5s}  {'Interpretation'}")
    print("-" * 110)
    for a1, a2 in key_pairs:
        valid = df[[f"{a1}_value", f"{a2}_value"]].dropna()
        if len(valid) < 10:
            continue
        r, pr = stats.pearsonr(valid[f"{a1}_value"], valid[f"{a2}_value"])
        rho, ps = stats.spearmanr(valid[f"{a1}_value"], valid[f"{a2}_value"])
        sig = "*" if pr < 0.001 else ("**" if pr < 0.01 else ("***" if pr < 0.05 else ""))
        # Interpretation
        if abs(r) < 0.1:
            interp = "negligible"
        elif abs(r) < 0.2:
            interp = "very weak" + sig
        elif abs(r) < 0.3:
            interp = "weak" + sig
        elif abs(r) < 0.5:
            interp = "moderate" + sig
        else:
            interp = "strong" + sig
        direction = "positive" if r > 0 else "negative"
        print(f"{a1:>12s} — {a2:<12s}  {r:+9.4f}  {pr:8.4f}  {rho:+10.4f}  {ps:8.4f}  {len(valid):5d}  {direction} {interp}")

    # ═══════════════════════════════════════════════════════════════════
    # 2. PARTIAL CORRELATION (controlling for days_from_surgery)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("2. PARTIAL CORRELATION (controlling for days_from_surgery)")
    print("=" * 70)

    df_surg = df[df["days_from_surgery"].notna()].copy()
    print(f"Samples with surgery timing: {len(df_surg)}")
    days = df_surg["days_from_surgery"].values

    print(f"\n{'Pair':>25s}  {'Raw r':>8s}  {'Partial r':>10s}  {'p_partial':>10s}  {'Δ':>8s}  {'Conclusion'}")
    print("-" * 85)

    partial_results = {}
    for a1, a2 in key_pairs:
        valid = df_surg[[f"{a1}_value", f"{a2}_value"]].dropna()
        if len(valid) < 20:
            continue

        x = valid[f"{a1}_value"].values
        y = valid[f"{a2}_value"].values
        idx = valid.index
        z = days[np.isin(df_surg.index, idx)]

        r_raw, _ = stats.pearsonr(x, y)
        r_partial, p_partial = partial_corr(x, y, z)

        delta = r_partial - r_raw
        if np.isnan(r_partial):
            continue

        if p_partial < 0.001:
            concl = "remains significant ***"
        elif p_partial < 0.01:
            concl = "remains significant **"
        elif p_partial < 0.05:
            concl = "remains significant *"
        elif abs(delta) > 0.05:
            concl = "explained by surgery timing"
        else:
            concl = "not significant"

        partial_results[(a1, a2)] = {
            "raw_r": float(r_raw), "partial_r": float(r_partial),
            "p_partial": float(p_partial), "delta": float(delta), "n": len(valid),
        }

        print(f"{a1:>12s} — {a2:<12s}  {r_raw:+8.4f}  {r_partial:+10.4f}  "
              f"{p_partial:10.5f}  {delta:+8.4f}  {concl}")

    # ═══════════════════════════════════════════════════════════════════
    # 3. STRATIFIED CORRELATION BY SURGERY PHASE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("3. STRATIFIED CORRELATION BY SURGERY PHASE")
    print("=" * 70)

    # Only for the key pairs of interest
    strat_pairs = [
        ("hemoglobin", "po2"),
        ("hemoglobin", "glucose"),
        ("hemoglobin", "lactate"),
        ("glucose", "po2"),
        ("glucose", "lactate"),
        ("lactate", "po2"),
    ]

    for a1, a2 in strat_pairs:
        print(f"\n--- {a1} — {a2} ---")
        print(f"  {'Phase':<22s}  {'Pearson r':>9s}  {'p':>8s}  {'Spearman ρ':>10s}  {'p':>8s}  {'n':>5s}")
        print(f"  {'-'*65}")

        for bin_name, lo, hi in SURGERY_BINS:
            mask = (df_surg["days_from_surgery"] >= lo) & (df_surg["days_from_surgery"] < hi)
            subset = df_surg.loc[mask, [f"{a1}_value", f"{a2}_value"]].dropna()
            if len(subset) < 8:
                print(f"  {bin_name:<22s}  {'--':>9s}  {'--':>8s}  {'--':>10s}  {'--':>8s}  {len(subset):5d} (insufficient)")
                continue

            r, pr = stats.pearsonr(subset[f"{a1}_value"], subset[f"{a2}_value"])
            rho, ps = stats.spearmanr(subset[f"{a1}_value"], subset[f"{a2}_value"])
            sig = " *" if pr < 0.05 else ""
            print(f"  {bin_name:<22s}  {r:+9.4f}  {pr:8.4f}  {rho:+10.4f}  {ps:8.4f}  {len(subset):5d}{sig}")

    # ═══════════════════════════════════════════════════════════════════
    # 4. SCATTER PLOTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("4. GENERATING SCATTER PLOTS")
    print("=" * 70)

    plot_pairs = [
        ("hemoglobin", "po2"),
        ("hemoglobin", "glucose"),
        ("hemoglobin", "lactate"),
        ("glucose", "po2"),
        ("glucose", "lactate"),
        ("lactate", "po2"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for i, (a1, a2) in enumerate(plot_pairs):
        ax = axes[i]
        valid = df[[f"{a1}_value", f"{a2}_value"]].dropna()
        x = valid[f"{a1}_value"].values
        y = valid[f"{a2}_value"].values

        r, pr = stats.pearsonr(x, y)
        rho, ps = stats.spearmanr(x, y)

        # Color by days_from_surgery if available
        idx = valid.index
        if "days_from_surgery" in df.columns:
            days_subset = df.loc[idx, "days_from_surgery"]
            has_days = days_subset.notna()
            # Only show colors where days available
            if has_days.sum() > 10:
                sc = ax.scatter(x[has_days], y[has_days],
                                c=days_subset[has_days].values,
                                cmap="coolwarm", alpha=0.5, s=15,
                                vmin=-10, vmax=14, edgecolors="none")
                # Also show no-days points in gray
                no_days = ~has_days
                if no_days.sum() > 0:
                    ax.scatter(x[no_days], y[no_days],
                               c="gray", alpha=0.2, s=10, edgecolors="none")
                cbar = plt.colorbar(sc, ax=ax)
                cbar.set_label("Days from surgery")
            else:
                ax.scatter(x, y, alpha=0.3, s=15, c="#1f77b4", edgecolors="none")
        else:
            ax.scatter(x, y, alpha=0.3, s=15, c="#1f77b4", edgecolors="none")

        # Regression line
        if len(x) > 5:
            from numpy.polynomial.polynomial import polyfit
            coeffs = polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = coeffs[0] + coeffs[1] * x_line
            ax.plot(x_line, y_line, "r-", linewidth=1.5, alpha=0.7)

        ax.set_xlabel(f"{a1} (corrected)")
        ax.set_ylabel(a2)
        ax.set_title(f"{a1} — {a2}\nr={r:.3f} (p={pr:.4f}), ρ={rho:.3f} (p={ps:.4f})",
                     fontsize=9)

    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "corrected_correlations.png"), dpi=150)
    plt.close()
    print("Saved corrected_correlations.png")

    # ═══════════════════════════════════════════════════════════════════
    # 5. SUMMARY: Comparison with Phase 1 (buggy) values
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("5. COMPARISON: Phase 1 (buggy) vs Phase 2 (corrected)")
    print("=" * 70)

    # Phase 1 buggy values (from original report)
    buggy = {
        ("hemoglobin", "po2"): -0.553,
        ("hemoglobin", "glucose"): -0.301,
        ("hemoglobin", "lactate"): 0.225,
        ("glucose", "po2"): -0.125,
        ("glucose", "lactate"): None,  # not reported
        ("lactate", "po2"): -0.207,
    }

    print(f"  {'Pair':>25s}  {'Buggy r':>9s}  {'Corrected r':>11s}  {'Δ':>8s}  {'Impact'}")
    print(f"  {'-'*70}")
    for (a1, a2), r_bug in buggy.items():
        valid = df[[f"{a1}_value", f"{a2}_value"]].dropna()
        r_corr, _ = stats.pearsonr(valid[f"{a1}_value"], valid[f"{a2}_value"])
        if r_bug is not None:
            delta = r_corr - r_bug
            if abs(delta) > 0.2:
                impact = "MAJOR — old result invalid"
            elif abs(delta) > 0.1:
                impact = "SIGNIFICANT change"
            elif abs(delta) > 0.03:
                impact = "Moderate change"
            else:
                impact = "Similar"
            print(f"  {a1:>12s} — {a2:<12s}  {r_bug:+9.4f}  {r_corr:+11.4f}  {delta:+8.4f}  {impact}")

    # Save matrices
    pearson.to_csv(os.path.join(OUTPUT_DIR, "pearson_corrected.csv"))
    spearman.to_csv(os.path.join(OUTPUT_DIR, "spearman_corrected.csv"))
    pd.DataFrame(partial_results).T.to_csv(os.path.join(OUTPUT_DIR, "partial_correlations.csv"))

    print(f"\nResults saved to {OUTPUT_DIR}/")
    return pearson, spearman, partial_results


if __name__ == "__main__":
    main()
