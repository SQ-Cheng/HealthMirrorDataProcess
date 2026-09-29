"""Analyze saved 24-hour five-fold OOF predictions by lab/video time gap."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp2_face_pretrained_head32_regression/outputs/5fold"
OUTPUT = HERE / "outputs"
EDGES = (0.0, 3.0, 6.0, 12.0, 24.0)
BIN_NAMES = ("0-3h", "3-6h", "6-12h", "12-24h")
KEY = ["fold", "target", "hospital_id", "video_id"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_joined():
    if not (SOURCE / "COMPLETE").is_file():
        raise RuntimeError(f"Five-fold OOF experiment is incomplete: {SOURCE}")
    manifest_path = SOURCE / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    targets = manifest["targets"]
    if manifest["folds"] != 5 or manifest["protocol"] != "face_regression":
        raise RuntimeError("Unexpected OOF source protocol")
    predictions_path = SOURCE / "oof_predictions.csv"
    predictions = pd.read_csv(predictions_path, dtype={"hospital_id": str, "video_id": str})
    if (not predictions["split"].eq("test").all()
            or set(predictions["target"]) != set(targets)
            or set(predictions["fold"]) != set(range(5))
            or predictions.duplicated(KEY).any()
            or predictions.duplicated(["target", "video_id"]).any()
            or not predictions["frame_count"].eq(20).all()):
        raise AssertionError("OOF predictions are not unique 20-frame held-out videos")

    parts = []
    source_hashes = {"experiment_manifest": sha256(manifest_path),
                     "oof_predictions": sha256(predictions_path), "splits": {}}
    for target in targets:
        for fold in range(5):
            path = SOURCE / "splits" / f"{target}_fold{fold}.csv"
            split = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
            test = split.loc[split["split"].eq("test"), [
                "hospital_id", "video_id", "raw_value", "match_delta_h",
                "match_signed_delta_h", "source_sample_id",
            ]].copy()
            test.insert(0, "target", target)
            test.insert(0, "fold", fold)
            parts.append(test)
            source_hashes["splits"][f"{target}_fold{fold}"] = sha256(path)
    matching = pd.concat(parts, ignore_index=True)
    if matching.duplicated(KEY).any() or len(matching) != len(predictions):
        raise AssertionError("Split/test cardinality or uniqueness differs from OOF")
    joined = predictions.merge(matching, on=KEY, how="outer", validate="one_to_one",
                               indicator=True)
    if not joined["_merge"].eq("both").all():
        raise AssertionError("OOF predictions do not exactly match saved test splits")
    joined = joined.drop(columns="_merge")
    for field in ("raw_value_x", "raw_value_y", "y_true", "y_pred",
                  "match_delta_h", "match_signed_delta_h"):
        if not np.isfinite(joined[field].to_numpy(float)).all():
            raise AssertionError(f"Non-finite {field} in OOF join")
    if (not np.allclose(joined["raw_value_x"], joined["raw_value_y"], rtol=0, atol=1e-5)
            or not np.allclose(joined["y_true"], joined["raw_value_y"], rtol=0, atol=1e-5)):
        raise AssertionError("Saved OOF targets disagree with matched lab values")
    if (not np.allclose(joined["match_delta_h"], joined["match_signed_delta_h"].abs(),
                        rtol=0, atol=1e-6)
            or not joined["match_delta_h"].between(0, 24, inclusive="both").all()):
        raise AssertionError("Match time difference is inconsistent or outside 24 hours")
    joined = joined.rename(columns={"raw_value_y": "matched_raw_value"})
    joined = joined.drop(columns="raw_value_x")
    joined["time_bin"] = pd.cut(joined["match_delta_h"], EDGES, labels=BIN_NAMES,
                                right=False, include_lowest=True).astype(str)
    joined.loc[joined["match_delta_h"].eq(24), "time_bin"] = BIN_NAMES[-1]
    if not joined["time_bin"].isin(BIN_NAMES).all():
        raise AssertionError("A test prediction was not assigned to a time bin")
    return joined, targets, source_hashes


def summarize(joined, targets):
    rows = []
    for target in targets:
        for bin_name in BIN_NAMES:
            group = joined.loc[joined["target"].eq(target) & joined["time_bin"].eq(bin_name)]
            n = len(group)
            truth = group["y_true"].to_numpy(float)
            prediction = group["y_pred"].to_numpy(float)
            sd = float(np.std(truth, ddof=1)) if n >= 2 else np.nan
            r = float(np.corrcoef(truth, prediction)[0, 1]) if (
                n >= 2 and np.std(truth) > 0 and np.std(prediction) > 0
            ) else np.nan
            rows.append({
                "target": target, "time_bin": bin_name,
                "n_videos": n, "n_patients": group["hospital_id"].nunique(),
                "pearson_r": r,
                "mae": mean_absolute_error(truth, prediction) if n else np.nan,
                "r2": r2_score(truth, prediction) if n >= 2 and sd > 0 else np.nan,
                "target_sd": sd,
                "mean_match_delta_h": group["match_delta_h"].mean() if n else np.nan,
            })
    return pd.DataFrame(rows)


def plot_metric(summary, targets, field, ylabel, filename, color):
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.2), constrained_layout=True,
                                sharex=True)
    if field == "pearson_r":
        observed = summary[field].dropna()
        padding = max(0.08, 0.12 * (observed.max() - observed.min()))
        r_limits = (max(-1, observed.min() - padding), min(1, observed.max() + padding))
    for axis, target in zip(axes.flat, targets):
        rows = summary.loc[summary["target"].eq(target)].set_index("time_bin").loc[list(BIN_NAMES)]
        values = rows[field].to_numpy(float)
        x = np.arange(len(BIN_NAMES))
        axis.plot(x, values, marker="o", linewidth=2, markersize=5, color=color)
        axis.set_title(TASK_LABELS.get(target, target), fontsize=11)
        axis.set_xticks(x, BIN_NAMES, fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        unit = TASK_UNITS.get(target, "") if field in ("mae", "target_sd") else ""
        axis.set_ylabel(f"{ylabel} ({unit})" if unit else ylabel)
        if field == "pearson_r":
            axis.set_ylim(*r_limits)
        if field in ("mae", "target_sd"):
            axis.set_ylim(bottom=0)
    figure.savefig(OUTPUT / "figures" / filename, dpi=180)
    plt.close(figure)


def plot_counts(summary, targets):
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.2), constrained_layout=True,
                                sharex=True)
    for axis, target in zip(axes.flat, targets):
        rows = summary.loc[summary["target"].eq(target)].set_index("time_bin").loc[list(BIN_NAMES)]
        counts = rows["n_videos"].to_numpy(int)
        bars = axis.bar(np.arange(4), counts, color="#3C7A89", width=0.62)
        axis.bar_label(bars, padding=2, fontsize=8)
        axis.set_title(TASK_LABELS.get(target, target), fontsize=11)
        axis.set_xticks(np.arange(4), BIN_NAMES, fontsize=8)
        axis.set_ylabel("OOF test videos")
        axis.set_ylim(0, max(counts) * 1.2 if max(counts) else 1)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(OUTPUT / "figures" / "video_counts.png", dpi=180)
    plt.close(figure)


def main():
    joined, targets, source_hashes = load_joined()
    summary = summarize(joined, targets)
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUTPUT / "oof_with_match_intervals.csv", index=False)
    summary.to_csv(OUTPUT / "metrics_by_time_bin.csv", index=False)
    for field, ylabel, filename, color in (
        ("pearson_r", "Pearson r", "pearson_r_by_time_bin.png", "#267D88"),
        ("mae", "MAE", "mae_by_time_bin.png", "#C15E4A"),
        ("r2", "R2", "r2_by_time_bin.png", "#637D39"),
        ("target_sd", "Target SD", "target_sd_by_time_bin.png", "#785D99"),
    ):
        plot_metric(summary, targets, field, ylabel, filename, color)
    plot_counts(summary, targets)
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps({
        "source": str(SOURCE), "source_sha256": source_hashes,
        "analysis_unit": "one video and matched lab value per target; pooled five-fold test OOF",
        "bin_boundaries_hours": list(EDGES),
        "bin_policy": "left-closed/right-open, except 24h included in final bin",
        "metric_scale": "original lab units (not robust-scaled)",
        "target_sd_ddof": 1,
        "n_predictions": len(joined), "targets": targets,
    }, indent=2), encoding="utf-8")
    print(f"[complete] {len(joined)} OOF video predictions; output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
