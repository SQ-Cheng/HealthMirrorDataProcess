"""Direction B: Cross-Analyte Correlation Analysis in Cardiac Surgery Context.

Analyzes how different lab analytes correlate with each other, clusters patients
by multi-analyte profiles, and examines whether ECG features differ across clusters.

Key analyses:
  1. Pearson/Spearman correlation matrix of 6 lab analytes
  2. Patient clustering by multi-analyte profile (K-means on normalized lab values)
  3. ANOVA: do ECG features differ significantly across lab-defined clusters?
  4. Multi-analyte risk score prediction from ECG features
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ..config import ANALYTES, OUTPUT_DIR, SEED


def run_cross_analyte_analysis(output_dir=None):
    """Run the full cross-analyte correlation analysis.

    Uses the feature dataset from Direction A (ecg_features.csv).
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_b")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    features_path = os.path.join(OUTPUT_DIR, "direction_a", "ecg_features.csv")
    if not os.path.exists(features_path):
        print("ERROR: ecg_features.csv not found. Run Direction A first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")

    # ── 1. Correlation Matrix ──
    print("\n" + "=" * 60)
    print("1. Cross-Analyte Correlation Matrix")
    print("=" * 60)

    value_cols = [f"{a}_value" for a in ANALYTES]
    corr_df = df[value_cols].dropna(how="all")

    # Pearson
    pearson_corr = corr_df.corr(method="pearson")
    # Rename for readability
    pearson_corr.index = ANALYTES
    pearson_corr.columns = ANALYTES

    # Spearman (more robust to outliers)
    spearman_corr = corr_df.corr(method="spearman")
    spearman_corr.index = ANALYTES
    spearman_corr.columns = ANALYTES

    print("\nPearson Correlation:")
    print(pearson_corr.to_string(float_format=".3f"))

    print("\nSpearman Correlation:")
    print(spearman_corr.to_string(float_format=".3f"))

    # Find significant correlations (p < 0.05)
    print("\nSignificant Pearson correlations (|r| > 0.1):")
    for i, a1 in enumerate(ANALYTES):
        for j, a2 in enumerate(ANALYTES):
            if i < j:
                vals = corr_df[[f"{a1}_value", f"{a2}_value"]].dropna()
                if len(vals) > 10:
                    r, p = stats.pearsonr(vals.iloc[:, 0], vals.iloc[:, 1])
                    if abs(r) > 0.1 and p < 0.05:
                        print(f"  {a1} — {a2}: r={r:.3f}, p={p:.4f}, n={len(vals)}")

    # Save correlation matrices
    pearson_corr.to_csv(os.path.join(output_dir, "pearson_correlation.csv"))
    spearman_corr.to_csv(os.path.join(output_dir, "spearman_correlation.csv"))

    # ── 2. Patient Clustering by Multi-Analyte Profile ──
    print("\n" + "=" * 60)
    print("2. Patient Clustering by Multi-Analyte Profile")
    print("=" * 60)

    # Aggregate by patient (median of each analyte)
    patient_data = df.groupby("hospital_id").agg({
        f"{a}_value": "median" for a in ANALYTES
    }).dropna(how="all")

    # For clustering, take patients with at least 3 non-NaN analytes
    patient_data = patient_data.dropna(thresh=3)
    print(f"Patients with >=3 analytes: {len(patient_data)}")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(patient_data.fillna(patient_data.median()))

    # Determine optimal K using elbow method
    inertias = []
    K_range = range(1, min(10, len(patient_data)))
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    # Choose K=3 for interpretability
    k_opt = 3
    km = KMeans(n_clusters=k_opt, random_state=SEED, n_init=10)
    clusters = km.fit_predict(X_scaled)
    patient_data["cluster"] = clusters

    print(f"\nCluster sizes (K={k_opt}):")
    for c in range(k_opt):
        c_size = (clusters == c).sum()
        print(f"  Cluster {c}: {c_size} patients")

        # Mean analyte profile
        c_data = patient_data[patient_data["cluster"] == c]
        print(f"    Mean values:")
        for a in ANALYTES:
            vals = c_data[f"{a}_value"].dropna()
            if len(vals) > 0:
                print(f"      {a}: {vals.mean():.2f} ± {vals.std():.2f}")

    # PCA visualization
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    axes[0].plot(list(K_range), inertias, "bo-")
    axes[0].set_xlabel("Number of clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("K-means Elbow Curve")
    axes[0].axvline(x=k_opt, color="r", linestyle="--", label=f"K={k_opt}")
    axes[0].legend()

    # PCA scatter
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for c in range(k_opt):
        mask = clusters == c
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors[c], label=f"Cluster {c}", alpha=0.6, s=30)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].set_title("Patient Clusters (PCA)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "patient_clusters.png"), dpi=150)
    plt.close()
    print(f"Saved patient_clusters.png")

    # ── 3. ECG Features vs Lab Clusters ──
    print("\n" + "=" * 60)
    print("3. ECG Features vs Lab-Defined Clusters")
    print("=" * 60)

    # Merge cluster info back to df
    df_with_cluster = df.merge(
        patient_data[["cluster"]], left_on="hospital_id", right_index=True, how="inner"
    )
    print(f"Samples with cluster label: {len(df_with_cluster)}")

    # ECG feature columns
    ecg_feat_cols = [
        "hr", "sdnn", "rmssd", "pnn50",
        "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
        "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
        "ecg_total_power", "ecg_lf_hf_ratio",
    ]

    # ANOVA / Kruskal-Wallis: which ECG features vary across clusters?
    print("\nECG features that differ across lab clusters (Kruskal-Wallis, p < 0.05):")
    significant_feats = []
    for feat in ecg_feat_cols:
        groups = [df_with_cluster[df_with_cluster["cluster"] == c][feat].dropna().values
                  for c in range(k_opt)]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                h, p = stats.kruskal(*groups)
                if p < 0.05:
                    means = [np.mean(g) for g in groups]
                    print(f"  {feat:30s}: H={h:.2f}, p={p:.4f}, means={[f'{m:.2f}' for m in means]}")
                    significant_feats.append(feat)
            except Exception:
                pass

    if not significant_feats:
        print("  No features significantly different across clusters.")

    # ── 4. Multi-Analyte Risk Score ──
    print("\n" + "=" * 60)
    print("4. Multi-Analyte Risk Score Prediction from ECG Features")
    print("=" * 60)

    # Define a composite risk score: normalized sum of abnormal indicators
    threshold_map = {
        "lactate": (2.0, "high"),
        "troponin": (34.0, "high"),
        "glucose": (7.8, "high"),
        "hemoglobin": (120.0, "low"),
        "po2": (80.0, "low"),
        "pco2": (50.0, "high"),
    }

    # Compute risk score for each sample
    risk_scores = []
    for _, row in df.iterrows():
        score = 0.0
        count = 0
        for a, (thresh, direction) in threshold_map.items():
            val = row[f"{a}_value"]
            if pd.notna(val):
                if direction == "high":
                    score += max(0, (val - thresh) / max(thresh, 0.01))
                else:
                    score += max(0, (thresh - val) / max(thresh, 0.01))
                count += 1
        risk_scores.append(score / max(count, 1))
    df["risk_score"] = risk_scores

    print(f"Risk score distribution: mean={np.mean(risk_scores):.3f}, "
          f"std={np.std(risk_scores):.3f}, "
          f"min={np.min(risk_scores):.3f}, max={np.max(risk_scores):.3f}")

    # Predict risk score from ECG features using Random Forest
    X_feat = df[ecg_feat_cols].fillna(df[ecg_feat_cols].median())
    X_feat = X_feat.replace([np.inf, -np.inf], 0.0)
    y_risk = df["risk_score"].dropna()

    # Align indices
    valid_idx = y_risk.index.intersection(X_feat.index)
    X_feat = X_feat.loc[valid_idx]
    y_risk = y_risk.loc[valid_idx]

    print(f"Valid samples for risk prediction: {len(y_risk)}")

    # Group K-Fold CV
    groups = df.loc[valid_idx, "hospital_id"].values
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)

    r2_scores = []
    mae_scores = []
    for train_idx, test_idx in gkf.split(X_feat, y_risk, groups):
        X_tr, X_te = X_feat.iloc[train_idx], X_feat.iloc[test_idx]
        y_tr, y_te = y_risk.iloc[train_idx], y_risk.iloc[test_idx]

        scaler_local = StandardScaler()
        X_tr_s = scaler_local.fit_transform(X_tr)
        X_te_s = scaler_local.transform(X_te)

        rf.fit(X_tr_s, y_tr)
        y_pred = rf.predict(X_te_s)

        r2_scores.append(r2_score(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))

    print(f"Risk score prediction (5-fold CV):")
    print(f"  R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"  MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")

    # Feature importance for risk score
    rf.fit(StandardScaler().fit_transform(X_feat), y_risk)
    feat_imp = sorted(zip(ecg_feat_cols, rf.feature_importances_),
                      key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 ECG features for risk score prediction:")
    for feat, imp in feat_imp[:5]:
        print(f"  {feat:30s}: {imp:.4f}")

    # ── Save ──
    df.to_csv(os.path.join(output_dir, "samples_with_risk.csv"), index=False)
    print(f"\nResults saved to {output_dir}/")

    return {
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "n_clusters": k_opt,
        "cluster_sizes": [int((clusters == c).sum()) for c in range(k_opt)],
        "significant_ecg_features": significant_feats,
        "risk_r2_mean": float(np.mean(r2_scores)),
        "risk_mae_mean": float(np.mean(mae_scores)),
    }


if __name__ == "__main__":
    run_cross_analyte_analysis()
