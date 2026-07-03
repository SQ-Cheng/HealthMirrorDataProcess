"""Direction A: Train & Evaluate Classical ML models on ECG features.

Compares multiple models:
  - Logistic Regression (L1, L2)
  - Random Forest
  - XGBoost
  - MLP (small neural network on features)

Evaluates on patient-level split to avoid data leakage.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ..config import OUTPUT_DIR, SEED

warnings.filterwarnings("ignore")

# Feature columns (all numeric ECG-derived features)
FEATURE_COLS = [
    "hr", "sdnn", "rmssd", "pnn50", "n_rr",
    "ecg_max", "ecg_min", "ecg_ptp", "ecg_rms",
    "ecg_skewness", "ecg_kurtosis", "ecg_zero_crossing_rate", "ecg_mean_abs",
    "ecg_total_power",
    "ecg_vlf_power", "ecg_lf_power", "ecg_hf_power",
    "ecg_vlf_norm", "ecg_lf_norm", "ecg_hf_norm",
    "ecg_lf_hf_ratio", "ecg_spectral_centroid",
]

TARGETS = [
    "lactate_abnormal",
    "troponin_abnormal",
    "glucose_abnormal",
    "po2_abnormal",
    "pco2_abnormal",
    "hemoglobin_low",
]

MODEL_CONFIGS = {
    "logistic_l2": lambda: LogisticRegression(
        penalty="l2", C=1.0, class_weight="balanced",
        max_iter=1000, random_state=SEED, solver="lbfgs"),
    "logistic_l1": lambda: LogisticRegression(
        penalty="l1", C=0.5, class_weight="balanced",
        max_iter=1000, random_state=SEED, solver="saga"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1),
    "xgboost": None,  # Will try to import
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu",
        alpha=0.01, max_iter=500, random_state=SEED, early_stopping=True),
}

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    MODEL_CONFIGS["xgboost"] = lambda: XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        scale_pos_weight=1, random_state=SEED, eval_metric="logloss",
        verbosity=0)
except ImportError:
    pass


def _evaluate_model(y_true, y_score, threshold=0.5):
    """Compute comprehensive metrics."""
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"balanced_accuracy": np.nan, "roc_auc": np.nan,
                "f1": np.nan, "accuracy": np.nan, "n": len(y_true),
                "positive_rate": float(np.mean(y_true)) if len(y_true) > 0 else np.nan}

    y_pred = (y_score >= threshold).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def _get_feature_matrix(features_df, feature_cols=FEATURE_COLS):
    """Extract feature matrix, imputing NaN with column median."""
    X = features_df[feature_cols].copy()
    # Impute NaN with column median
    X = X.fillna(X.median())
    # Replace infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    return X.to_numpy(dtype=np.float32)


def train_and_evaluate(features_df, output_dir=None):
    """Train all models on all targets, evaluate with patient-level split.

    Returns:
        results_df: DataFrame of all metrics
        feature_importance_df: DataFrame of feature importances
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "direction_a")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    all_importances = []

    # Prepare data
    X_all = _get_feature_matrix(features_df)
    hospital_ids = features_df["hospital_id"].values.astype(str)
    print(f"Feature matrix: {X_all.shape}")

    # For each target
    for target in TARGETS:
        y_all = pd.to_numeric(features_df[target], errors="coerce").values
        valid = np.isfinite(y_all)
        X_valid = X_all[valid]
        y_valid = y_all[valid]
        h_valid = hospital_ids[valid]

        n_pos = int(np.sum(y_valid > 0.5))
        n_neg = int(np.sum(y_valid < 0.5))
        print(f"\n{'='*60}")
        print(f"Target: {target}  (pos={n_pos}, neg={n_neg}, total={len(y_valid)})")

        if n_pos < 5 or n_neg < 5:
            print(f"  SKIP: insufficient samples")
            all_results.append({
                "target": target, "model": "all",
                "balanced_accuracy": np.nan, "roc_auc": np.nan,
                "reason": f"insufficient: pos={n_pos}, neg={n_neg}",
            })
            continue

        # Patient-level split
        unique_hids = np.unique(h_valid)
        if len(unique_hids) < 5:
            print(f"  SKIP: too few patients")
            continue

        # Stratified split by patient
        # Get patient-level labels (majority vote)
        patient_labels = {}
        for hid, y_val in zip(h_valid, y_valid):
            patient_labels[hid] = patient_labels.get(hid, []) + [y_val]
        patient_y = np.array([np.mean(v) > 0.5 for v in patient_labels.values()])
        patient_ids = np.array(list(patient_labels.keys()))

        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
            train_pidx, test_pidx = next(sss.split(patient_ids, patient_y))
        except ValueError:
            # Fallback to random split
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(patient_ids))
            n_test = max(1, int(len(patient_ids) * 0.30))
            train_pidx = perm[:len(patient_ids) - n_test]
            test_pidx = perm[len(patient_ids) - n_test:]

        train_patients = set(patient_ids[train_pidx])
        test_patients = set(patient_ids[test_pidx])

        train_mask = np.array([hid in train_patients for hid in h_valid])
        test_mask = np.array([hid in test_patients for hid in h_valid])

        X_train, y_train = X_valid[train_mask], y_valid[train_mask]
        X_test, y_test = X_valid[test_mask], y_valid[test_mask]

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Train positive rate: {np.mean(y_train):.2%}")
        print(f"  Test positive rate: {np.mean(y_test):.2%}")

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train each model
        for model_name, model_fn in MODEL_CONFIGS.items():
            if model_fn is None:
                continue

            try:
                model = model_fn()
                model.fit(X_train_scaled, y_train)

                # Predict
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_score = model.decision_function(X_test_scaled)

                metrics = _evaluate_model(y_test, y_score)
                result = {"target": target, "model": model_name, **metrics}
                all_results.append(result)

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_).flatten()
                    if len(importances) != len(FEATURE_COLS):
                        importances = np.abs(model.coef_[0])
                else:
                    importances = np.zeros(len(FEATURE_COLS))

                for feat, imp in zip(FEATURE_COLS, importances):
                    all_importances.append({
                        "target": target, "model": model_name,
                        "feature": feat, "importance": float(imp),
                    })

                bacc = metrics.get("balanced_accuracy", np.nan)
                auc = metrics.get("roc_auc", np.nan)
                if isinstance(bacc, float):
                    print(f"  {model_name:20s}: bACC={bacc:.3f}  AUC={auc:.3f}  "
                          f"F1={metrics.get('f1', np.nan):.3f}")

            except Exception as e:
                print(f"  {model_name:20s}: ERROR - {e}")
                all_results.append({
                    "target": target, "model": model_name,
                    "balanced_accuracy": np.nan, "roc_auc": np.nan,
                    "reason": str(e),
                })

    # ── Compile results ──
    results_df = pd.DataFrame(all_results)
    importance_df = pd.DataFrame(all_importances)

    results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY: Classical ML on ECG Features")
    print(f"{'='*60}")

    # Best model per target
    valid_results = results_df[results_df["balanced_accuracy"].notna()]
    best_per_target = valid_results.loc[
        valid_results.groupby("target")["balanced_accuracy"].idxmax()
    ]

    print(f"\nBest results per target:")
    for _, r in best_per_target.iterrows():
        print(f"  {r['target']:25s} | {r['model']:15s} | "
              f"bACC={float(r['balanced_accuracy']):.3f}  "
              f"AUC={float(r['roc_auc']):.3f}  "
              f"F1={float(r['f1']):.3f}  "
              f"n={int(r['n'])}")

    # Macro average
    macro_bacc = best_per_target["balanced_accuracy"].mean()
    macro_auc = best_per_target["roc_auc"].mean()
    print(f"\nMacro bACC: {macro_bacc:.4f}")
    print(f"Macro AUC:  {macro_auc:.4f}")

    # Compare with Exp2 DL results
    print(f"\nComparison with Exp2 DL (bACC):")
    exp2_results = {
        "lactate_abnormal": 0.500,  # lactate_high
        "troponin_abnormal": 0.437,  # troponin_high
        "glucose_abnormal": 0.566,  # glucose_high
        "po2_abnormal": 0.495,  # po2_low
        "pco2_abnormal": 0.460,  # pco2_abnormal
        "hemoglobin_low": 0.273,  # hemoglobin_low
    }
    for _, r in best_per_target.iterrows():
        target = r["target"]
        dl_bacc = exp2_results.get(target, np.nan)
        ml_bacc = float(r["balanced_accuracy"])
        diff = ml_bacc - dl_bacc if not np.isnan(dl_bacc) else np.nan
        winner = "ML" if diff > 0 else ("DL" if diff < 0 else "tie")
        if not np.isnan(diff):
            print(f"  {target:25s}: ML={ml_bacc:.3f}  DL={dl_bacc:.3f}  "
                  f"Δ={diff:+.3f}  ({winner})")

    # Top features
    print(f"\nTop 5 most important features (averaged across targets):")
    avg_importance = importance_df.groupby("feature")["importance"].mean().sort_values(ascending=False)
    for feat, imp in avg_importance.head(5).items():
        print(f"  {feat:30s}: {imp:.4f}")

    return results_df, importance_df


if __name__ == "__main__":
    from .extract_features import build_feature_dataset

    features_df = build_feature_dataset()
    results_df, importance_df = train_and_evaluate(features_df)
