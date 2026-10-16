"""Incremental same-test comparisons with patient-cluster paired bootstrap CIs."""

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score

from study.common.run_video_loss_12h import sha256
from study.exp2_face_architecture_ablation.plots import panels, plot_one, LABELS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from . import config


NAMES = {"baseline": "Baseline", "a01_regularized_rgb_prior": "Regularized", "a02_video_patient_balance": "Video balance",
    "a03_hierarchical_rgb_warmstart": "RGB warm start", "a04_log_density_rank": "Log + rank",
    "a05_color_safe_views3": "Color-safe", "a06_time_weighted": "Time-weighted", "a07_ir_tail_adaptation": "IR tail"}
COLORS = ("#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#998A00")


def plot_one_variant(name):
    LABELS[name] = NAMES[name]
    plot_one(config.OUTPUT / name, "regression", name, targets=config.TARGETS, loss_label="Original-frame SmoothL1")


def read_test(root, target):
    return pd.read_csv(root / f"runs/{target}/video_predictions.csv", dtype={"hospital_id": str, "video_id": str},
                       float_precision="round_trip").query("split == 'test'").sort_values("video_id").reset_index(drop=True)


def correlation(left, right):
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def cluster_bootstrap(baseline, candidate, seed, repeats=500):
    identity = ["hospital_id", "video_id"]
    pd.testing.assert_frame_equal(baseline[identity], candidate[identity])
    np.testing.assert_allclose(baseline.y_true, candidate.y_true, rtol=1e-12, atol=1e-12)
    patients = baseline.hospital_id.unique()
    positions = [np.flatnonzero(baseline.hospital_id.eq(patient)) for patient in patients]
    rng = np.random.default_rng(seed)
    y, old, new = baseline.y_true.to_numpy(), baseline.y_pred.to_numpy(), candidate.y_pred.to_numpy()
    dr, dm = [], []
    for _ in range(repeats):
        selected = np.concatenate([positions[i] for i in rng.integers(0, len(patients), size=len(patients))])
        dr.append(correlation(y[selected], new[selected])-correlation(y[selected], old[selected]))
        dm.append(np.abs(new[selected]-y[selected]).mean()-np.abs(old[selected]-y[selected]).mean())
    finite = np.asarray(dr)[np.isfinite(dr)]
    low_r, high_r = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return {"delta_r": correlation(y, new)-correlation(y, old), "delta_r_low": low_r, "delta_r_high": high_r,
            "delta_mae": np.abs(new-y).mean()-np.abs(old-y).mean(),
            "delta_mae_low": np.quantile(dm, .025), "delta_mae_high": np.quantile(dm, .975),
            "bootstrap_unit": "hospital_id cluster, all corresponding videos retained", "bootstrap_repeats": repeats}


def update_comparisons():
    from .run import completed_variant
    variants = [name for name in config.VARIANTS if completed_variant(name)]
    if not variants:
        return
    names = ["baseline", *variants]
    base = config.BASELINE / f"regression/{config.ARCHITECTURE}"
    roots = {"baseline": base, **{name: config.OUTPUT / name for name in variants}}
    scalers = json.loads((config.BASELINE / "target_scalers.json").read_text())
    figures, tables = config.OUTPUT / "figures", config.OUTPUT / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    metrics, audits, deltas, val_scores = [], [], [], []
    for target in config.TARGETS:
        reference = read_test(base, target)
        for name, root in roots.items():
            candidate = read_test(root, target)
            pd.testing.assert_frame_equal(reference[["hospital_id", "video_id"]], candidate[["hospital_id", "video_id"]])
            np.testing.assert_allclose(reference.y_true, candidate.y_true, rtol=1e-12, atol=1e-12)
            if not candidate.frame_count.eq(20).all() or not np.isfinite(candidate[["y_true", "y_pred"]]).all().all():
                raise RuntimeError("Comparison has missing/nonfinite test data")
            frame = pd.read_csv(root / f"runs/{target}/metrics.csv")
            row = frame.loc[frame.split.eq("test")].iloc[0].to_dict()
            row.update(model=name, target=target, explained_variance=explained_variance_score(candidate.y_true, candidate.y_pred),
                       mae_reference_iqr=row["mae"]/scalers[target]["iqr"])
            metrics.append(row)
            val_scores.append({"model": name, "target": target, "val_mae": frame.loc[frame.split.eq("val"), "mae"].iloc[0]})
            audits.append({"model": name, "target": target, "test_videos": len(candidate),
                "test_patients": candidate.hospital_id.nunique(), "same_labels_split_and_twenty_frame_pairs": True,
                "warmstart_extra_rgb_training_data": name in ("a03_hierarchical_rgb_warmstart", "a07_ir_tail_adaptation")})
            if name != "baseline":
                # Only compute a new variant's intervals; reuse unchanged completed ones.
                cache = config.OUTPUT / name / "paired_bootstrap.csv"
                expected_hashes = {"baseline_predictions_sha256": sha256(base / f"runs/{target}/video_predictions.csv"),
                                   "candidate_predictions_sha256": sha256(root / f"runs/{target}/video_predictions.csv")}
                if cache.exists():
                    previous = pd.read_csv(cache)
                    found = previous.loc[previous.target.eq(target)]
                    if len(found) and all(key in found and found.iloc[0][key] == value for key, value in expected_hashes.items()):
                        deltas.append(found.iloc[0].to_dict())
                        continue
                seed = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")
                result = cluster_bootstrap(reference, candidate, seed)
                deltas.append({"model": name, "target": target, **result, **expected_hashes, "reference_iqr": scalers[target]["iqr"]})
    table = pd.DataFrame(metrics)
    table.to_csv(tables / "matched_test_metrics.csv", index=False)
    pd.DataFrame(audits).to_csv(tables / "test_cohort_audit.csv", index=False)
    ci = pd.DataFrame(deltas)
    ci.to_csv(tables / "paired_patient_bootstrap.csv", index=False)
    for name in variants:
        ci.loc[ci.model.eq(name)].to_csv(config.OUTPUT / name / "paired_bootstrap.csv", index=False)
    val = pd.DataFrame(val_scores)
    winners = val.sort_values("val_mae").groupby("target", sort=False).head(1)
    winners.to_csv(tables / "best_by_validation_only.csv", index=False)
    macro = table.groupby("model", sort=False)[["mae_reference_iqr", "pearson_r", "r2"]].mean().reindex(names)
    macro.to_csv(tables / "macro_test_metrics.csv")
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("pearson_r", "Pearson r"), ("r2", "$R^2$"), ("explained_variance", "Explained variance")):
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            subset = table.loc[table.target.eq(target)].set_index("model").loc[names]
            bars = axis.bar(range(len(names)), subset[metric], color=COLORS[:len(names)])
            axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=3)
            axis.set_xticks(range(len(names)), [NAMES[name] for name in names], rotation=45, ha="right", fontsize=7)
            axis.set(title=TASK_LABELS[target], ylabel=label+(f" ({TASK_UNITS[target]})" if metric in ("mae", "rmse") else ""))
            axis.grid(axis="y", alpha=.2)
            axis.axhline(0, color=".5", linewidth=.5)
            axis.margins(y=.2)
        figure.suptitle(f"Same fixed patient splits / paired20 test data | {label}")
        figure.supxlabel("Exploratory fixed protocols; RGB warm starts use additional original RGB training videos. No test-based configuration selection.", fontsize=9)
        figure.savefig(figures / f"{metric}_comparison.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, key, label in zip(axes, ("mae_reference_iqr", "pearson_r"), ("Macro MAE / baseline train IQR", "Macro mean Pearson r")):
        bars = axis.bar(range(len(names)), macro[key], color=COLORS[:len(names)])
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set_xticks(range(len(names)), [NAMES[name] for name in names], rotation=35, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.2)
    figure.savefig(figures / "macro_overview.png", dpi=180)
    plt.close(figure)
    for name in variants:
        subset = ci.loc[ci.model.eq(name)].set_index("target").loc[list(config.TARGETS)]
        figure, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        for axis, stem, label in zip(axes, ("delta_r", "delta_mae"), ("Delta Pearson r (positive better)", "Delta MAE / baseline IQR (negative better)")):
            divisor = subset.reference_iqr.to_numpy() if stem == "delta_mae" else np.ones(len(subset))
            values = subset[stem].to_numpy()/divisor
            low, high = subset[stem+"_low"].to_numpy()/divisor, subset[stem+"_high"].to_numpy()/divisor
            positions = np.arange(len(subset))
            axis.barh(positions, values, color="#0072B2", alpha=.75)
            # Bootstrap percentile intervals need not contain the point estimate.
            axis.hlines(positions, low, high, color=".2", linewidth=1)
            axis.scatter(values, positions, color=".2", s=12, zorder=3)
            axis.set_yticks(positions, [TASK_LABELS[target] for target in config.TARGETS])
            axis.invert_yaxis()
            axis.axvline(0, color=".5", linestyle="--", linewidth=.8)
            axis.set_xlabel(label)
        figure.suptitle(NAMES[name]+" vs baseline | paired patient-cluster 95% bootstrap intervals")
        figure.savefig(config.OUTPUT / name / "figures/paired_improvement_ci.png", dpi=180)
        plt.close(figure)
    print(f"[comparison-updated] {len(variants)} ablations complete; {figures}", flush=True)


if __name__ == "__main__":
    update_comparisons()
