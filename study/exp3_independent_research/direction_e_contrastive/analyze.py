"""Direction E: Patient Similarity via ECG Feature Embeddings.

A lightweight approach to patient representation learning:
  1. Use extracted ECG features as a patient "fingerprint"
  2. Compute patient-patient similarity matrix
  3. Test: are patients with similar ECG features also similar in lab values?
  4. Test: can we identify the same patient across different sessions?

This avoids expensive contrastive pre-training and leverages Direction A's features.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import ANALYTES, OUTPUT_DIR, SEED


def run_patient_similarity_analysis(output_dir=None):
    """Analyze patient similarity using ECG features."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_e")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    features_path = os.path.join(OUTPUT_DIR, "direction_a", "ecg_features.csv")
    if not os.path.exists(features_path):
        print("ERROR: ecg_features.csv not found. Run Direction A first.")
        return

    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")

    ecg_feat_cols = [
        "hr", "sdnn", "rmssd", "pnn50",
        "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
        "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
        "ecg_total_power", "ecg_lf_hf_ratio",
    ]

    # Impute and standardize
    X = df[ecg_feat_cols].copy()
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], 0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 1. Patient-Level Aggregation ──
    print("\n" + "=" * 60)
    print("1. Patient-Level ECG Profiles")
    print("=" * 60)

    # Aggregate by patient: median ECG feature per patient
    patient_ecg = df.groupby("hospital_id")[ecg_feat_cols].median()
    print(f"Unique patients with ECG: {len(patient_ecg)}")

    # Aggregate lab values per patient
    patient_lab = df.groupby("hospital_id")[[f"{a}_value" for a in ANALYTES]].median()

    # ── 2. Patient-Patient Similarity ──
    print("\n" + "=" * 60)
    print("2. ECG Similarity vs Lab Similarity")
    print("=" * 60)

    # Compute ECG distance matrix
    X_patient = patient_ecg.fillna(patient_ecg.median()).values
    X_patient = StandardScaler().fit_transform(X_patient)

    ecg_dist = squareform(pdist(X_patient, metric="euclidean"))

    # Compute lab distance matrix (for patients with lab data)
    common_patients = patient_ecg.index.intersection(patient_lab.index)
    print(f"Patients with both ECG and lab data: {len(common_patients)}")

    if len(common_patients) >= 10:
        common_idx = {pid: i for i, pid in enumerate(patient_ecg.index)
                      if pid in common_patients}
        common_list = [pid for pid in patient_ecg.index if pid in common_patients]

        # Get ECG and lab values for common patients
        ecg_common = np.array([X_patient[patient_ecg.index.get_loc(pid)]
                               for pid in common_list])
        lab_common = patient_lab.loc[common_list].fillna(
            patient_lab.loc[common_list].median()).values
        lab_common = StandardScaler().fit_transform(lab_common)

        ecg_dist_common = squareform(pdist(ecg_common, metric="euclidean"))
        lab_dist_common = squareform(pdist(lab_common, metric="euclidean"))

        # Correlation: ECG distance vs lab distance
        ecg_upper = ecg_dist_common[np.triu_indices_from(ecg_dist_common, k=1)]
        lab_upper = lab_dist_common[np.triu_indices_from(lab_dist_common, k=1)]

        r, p = spearmanr(ecg_upper, lab_upper)
        print(f"Spearman correlation (ECG dist vs Lab dist): ρ={r:.4f}, p={p:.4f}")
        print(f"  Interpretation: {'Patients with similar ECG also have similar lab values' if r > 0 and p < 0.05 else 'No significant relationship'}")

        # Also test: Pearson
        from scipy.stats import pearsonr
        rp, pp = pearsonr(ecg_upper, lab_upper)
        print(f"Pearson correlation: r={rp:.4f}, p={pp:.4f}")

    # ── 3. Same-Patient vs Different-Patient ECG Similarity ──
    print("\n" + "=" * 60)
    print("3. Intra-Patient vs Inter-Patient ECG Similarity")
    print("=" * 60)

    # For patients with multiple sessions, compute intra-patient ECG distance
    session_counts = df.groupby("hospital_id").size()
    multi_session_patients = session_counts[session_counts >= 2].index
    print(f"Patients with >=2 sessions: {len(multi_session_patients)}")

    intra_distances = []
    inter_distances = []

    for pid in multi_session_patients[:50]:  # Sample for efficiency
        pid_sessions = df[df["hospital_id"] == pid]
        if len(pid_sessions) < 2:
            continue

        pid_features = pid_sessions[ecg_feat_cols].fillna(X.median()).values
        pid_features = scaler.transform(pid_features)

        # Intra-patient: all pairs within same patient
        for i in range(len(pid_features)):
            for j in range(i + 1, len(pid_features)):
                intra_distances.append(np.linalg.norm(pid_features[i] - pid_features[j]))

        # Inter-patient: pairs with different patients
        other_pids = [op for op in multi_session_patients[:50] if op != pid]
        for op in other_pids[:5]:
            op_sessions = df[df["hospital_id"] == op]
            op_features = op_sessions[ecg_feat_cols].fillna(X.median()).values
            op_features = scaler.transform(op_features)
            for pf in pid_features[:3]:
                for of in op_features[:3]:
                    inter_distances.append(np.linalg.norm(pf - of))

    intra_mean = np.mean(intra_distances) if intra_distances else np.nan
    inter_mean = np.mean(inter_distances) if inter_distances else np.nan

    print(f"Intra-patient ECG distance (mean): {intra_mean:.3f}")
    print(f"Inter-patient ECG distance (mean): {inter_mean:.3f}")

    if intra_distances and inter_distances:
        from scipy.stats import mannwhitneyu
        u, p_val = mannwhitneyu(intra_distances, inter_distances, alternative="less")
        ratio = inter_mean / max(intra_mean, 1e-6)
        print(f"Ratio (inter/intra): {ratio:.2f}x")
        print(f"Mann-Whitney U test (intra < inter): p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  ✓ Same patient sessions are significantly more similar in ECG features")
        else:
            print(f"  ✗ No significant difference between intra and inter patient ECG similarity")

    # ── 4. t-SNE Visualization ──
    print("\n" + "=" * 60)
    print("4. t-SNE Visualization of Patient ECG Profiles")
    print("=" * 60)

    if len(patient_ecg) >= 10:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(patient_ecg) - 1))
        X_tsne = tsne.fit_transform(X_patient)

        # Color by median lactate
        if "lactate_value" in patient_lab.columns:
            lactate_vals = patient_lab["lactate_value"].reindex(patient_ecg.index)
        else:
            lactate_vals = pd.Series(np.nan, index=patient_ecg.index)

        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                        c=lactate_vals.fillna(lactate_vals.median()),
                        cmap="RdYlBu_r", alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="Median Lactate (mmol/L)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title("Patient ECG Profiles (t-SNE) — colored by Lactate")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "patient_tsne_lactate.png"), dpi=150)
        plt.close()
        print("Saved patient_tsne_lactate.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary: Patient Similarity Analysis")
    print("=" * 60)
    print(f"  ECG-vs-Lab distance correlation: ρ={r:.4f}" if common_list else "  Insufficient data for lab similarity")
    print(f"  Intra/inter patient ECG distance ratio: {ratio:.2f}x" if intra_distances else "")

    results = {
        "ecg_lab_distance_spearman": float(r) if common_list else np.nan,
        "ecg_lab_distance_p": float(p) if common_list else np.nan,
        "intra_ecg_distance_mean": float(intra_mean),
        "inter_ecg_distance_mean": float(inter_mean),
        "n_patients_with_ecg": len(patient_ecg),
        "n_patients_with_lab": len(common_patients) if common_list else 0,
    }

    return results


if __name__ == "__main__":
    run_patient_similarity_analysis()
