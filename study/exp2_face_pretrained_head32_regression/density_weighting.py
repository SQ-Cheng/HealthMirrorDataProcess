"""Training-only inverse-density weights for the saved 20-frame task records."""

import numpy as np


BIN_COUNT = 12
TAIL_QUANTILES = (0.01, 0.99)
PSEUDOCOUNT_FRACTION = 0.005
MAX_WEIGHT_RATIO = 4.0


def make_density_weights(train_records):
    if set(train_records["split"]) != {"train"}:
        raise ValueError("Density weights must be fitted on training records only")
    values = train_records["robust_scaled_raw_value"].to_numpy(np.float64)
    if len(values) < BIN_COUNT or not np.isfinite(values).all():
        raise ValueError("Insufficient finite training values for density weighting")
    lower, upper = np.quantile(values, TAIL_QUANTILES)
    if upper <= lower:
        raise ValueError("Training values have no usable density range")
    edges = np.linspace(lower, upper, BIN_COUNT + 1)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, BIN_COUNT - 1)
    counts = np.bincount(bins, minlength=BIN_COUNT)
    smoothed = counts.astype(np.float64) + max(
        1.0, PSEUDOCOUNT_FRACTION * len(values)
    )
    inverse_density = np.sqrt(smoothed.max() / smoothed[bins])
    inverse_density = np.clip(inverse_density, 1.0, MAX_WEIGHT_RATIO)
    weights = inverse_density / inverse_density.mean()
    bin_weights = np.array([
        float(weights[bins == index][0]) if counts[index] else None
        for index in range(BIN_COUNT)
    ], dtype=object)
    audit = {
        "policy": "train_only_equal_width_inverse_sqrt_density",
        "value_column": "robust_scaled_raw_value",
        "frame_policy": "20 frames and 5 views per video; video and frame frequencies coincide",
        "bins": BIN_COUNT,
        "tail_quantiles": list(TAIL_QUANTILES),
        "pseudocount_fraction": PSEUDOCOUNT_FRACTION,
        "max_pre_normalization_weight_ratio": MAX_WEIGHT_RATIO,
        "bin_edges_scaled": edges.tolist(),
        "training_videos_per_bin": counts.astype(int).tolist(),
        "normalized_weight_per_bin": bin_weights.tolist(),
        "training_videos": int(len(values)),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "effective_video_count": float(weights.sum() ** 2 / np.square(weights).sum()),
    }
    return weights.astype(np.float32), audit
