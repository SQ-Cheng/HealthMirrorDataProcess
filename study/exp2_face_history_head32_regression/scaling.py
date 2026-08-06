"""Leakage-safe train-only robust scaling for raw regression targets."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustTargetScaler:
    target: str
    unit: str
    median: float
    q1: float
    q3: float
    iqr: float
    train_videos: int
    train_video_ids_sha256: str
    quantile_method: str = "linear"

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return (values - self.median) / self.iqr

    def inverse_transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values * self.iqr + self.median

    def to_dict(self):
        return asdict(self)


def fit_robust_target_scaler(target, records, unit):
    train = records.loc[records["split"].eq("train")].copy()
    if train.empty:
        raise ValueError(f"No training rows available to fit scaler for {target}")
    values = train["raw_value"].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite training raw values for {target}")
    q1, median, q3 = np.quantile(values, (0.25, 0.50, 0.75), method="linear")
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0.0:
        raise ValueError(f"Non-positive training IQR for {target}: {iqr}")
    video_ids = "\n".join(sorted(train["video_id"].astype(str))).encode()
    return RobustTargetScaler(
        target=target,
        unit=unit,
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=iqr,
        train_videos=len(train),
        train_video_ids_sha256=hashlib.sha256(video_ids).hexdigest(),
    )


def write_target_scalers(scalers, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "method": "train-only median/IQR robust scaling",
                "formula": "scaled=(raw_value-train_median)/train_IQR",
                "inverse_formula": "raw_value=scaled*train_IQR+train_median",
                "clipping": None,
                "targets": {
                    target: scaler.to_dict() for target, scaler in scalers.items()
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def apply_train_range_weights(records, bins=5, max_weight=4.0):
    """Fit fixed-width raw-value bins on train only and attach stable weights."""
    if bins < 2:
        raise ValueError("Range weighting requires at least two bins")
    result = records.copy()
    train_mask = result["split"].eq("train")
    train_values = result.loc[train_mask, "raw_value"].to_numpy(np.float64)
    if not len(train_values) or not np.isfinite(train_values).all():
        raise ValueError("Range weighting requires finite training raw values")
    lower, upper = np.quantile(train_values, (0.01, 0.99), method="linear")
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(train_values.min()), float(train_values.max())
    if upper <= lower:
        raise ValueError("Range weighting requires non-constant training values")
    edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)
    train_bins = np.clip(np.searchsorted(edges[1:-1], train_values, side="right"), 0, bins - 1)
    counts = np.bincount(train_bins, minlength=bins)
    nonempty = counts > 0
    raw_weights = np.zeros(bins, dtype=np.float64)
    raw_weights[nonempty] = np.sqrt(len(train_values) / counts[nonempty])
    mean_weight = np.average(raw_weights[train_bins])
    raw_weights /= mean_weight
    raw_weights = np.minimum(raw_weights, float(max_weight))
    raw_weights /= np.average(raw_weights[train_bins])

    all_values = result["raw_value"].to_numpy(np.float64)
    all_bins = np.clip(np.searchsorted(edges[1:-1], all_values, side="right"), 0, bins - 1)
    result["range_bin"] = all_bins.astype(np.int16)
    result["range_weight"] = 1.0
    result.loc[train_mask, "range_weight"] = raw_weights[all_bins[train_mask]]
    policy = {
        "fit_scope": "training videos only",
        "value_space": "raw laboratory units",
        "binning": "five equal-width bins between train p01 and p99; tails assigned to edge bins",
        "bin_edges": edges.tolist(),
        "train_video_counts": counts.astype(int).tolist(),
        "bin_weights": raw_weights.tolist(),
        "weight_formula": "sqrt(train_video_count / bin_video_count), normalized to train mean 1",
        "maximum_weight_before_final_normalization": float(max_weight),
        "validation_test_weight": 1.0,
    }
    return result, policy
