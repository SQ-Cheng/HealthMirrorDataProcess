"""Incremental PNGs; separate fold means/SD and pooled out-of-fold estimates."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

from study.common.run_video_loss_12h import sha256
from study.exp2_face_architecture_ablation.plots import panels
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS
from study.exp2_sota_ablation.plots import cluster_bootstrap, correlation, COLORS
from . import config

METRICS = ("mae", "rmse", "pearson_r", "r2", "explained_variance")


def summarize(y, pred):
    return {"mae": mean_absolute_error(y, pred), "rmse": np.sqrt(mean_squared_error(y, pred)),
        "pearson_r": correlation(y, pred), "r2": r2_score(y, pred),
        "explained_variance": explained_variance_score(y, pred)}


def pooled_frame(name, target):
    frames = []
    for fold in range(5):
        path = config.OUTPUT/name/target/f"fold{fold}/video_predictions.csv"
        frame = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
        frames.append(frame.loc[frame.split.eq("test")])
    data = pd.concat(frames, ignore_index=True).sort_values("video_id").reset_index(drop=True)
    if data.video_id.duplicated().any() or data.groupby("hospital_id").fold.nunique().gt(1).any():
        raise RuntimeError("OOF video duplication or patient leakage")
    from study.exp2_sota_ablation.data import read_records
    expected = read_records(target).sort_values("video_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(data[["hospital_id", "video_id"]], expected[["hospital_id", "video_id"]])
    np.testing.assert_allclose(data.y_true, expected.raw_value, rtol=1e-12, atol=1e-12)
    if not data.frame_count.eq(20).all() or not np.isfinite(data[["y_true", "y_pred"]]).all().all():
        raise RuntimeError("Missing/nonfinite OOF frames")
    return data


def plot_history(root):
    for scope in ("selection", "refit"):
        files = [root/scope/f"{stage}_history.csv" for stage in ("frozen", "ir_tail")]
        frames = [pd.read_csv(file) for file in files if file.exists() and file.stat().st_size > 2]
        if not frames:
            continue
        history = pd.concat(frames, ignore_index=True)
        x = np.arange(len(history))
        figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        twin = axis.twinx()
        for column, label, color, style in (("train_mae", "Train MAE", "#0072B2", "-"),
                ("val_mae", "Inner val MAE", "#D55E00", "--")):
            if column in history:
                axis.plot(x, history[column], style, color=color, label=label)
        for column, label, color, style in (("train_pearson_r", "Train r", "#009E73", "-"),
                ("val_pearson_r", "Inner val r", "#CC79A7", "--")):
            if column in history:
                twin.plot(x, history[column], style, color=color, label=label, alpha=.85)
        correlations = history[[key for key in ("train_pearson_r", "val_pearson_r") if key in history]].to_numpy().ravel()
        finite = correlations[np.isfinite(correlations)]
        if len(finite):
            margin = max(.03, np.ptp(finite)*.1)
            twin.set_ylim(max(-1.05, finite.min()-margin), min(1.05, finite.max()+margin))
        axis.set(xlabel="Epoch (concatenated stages)", ylabel="MAE, original units", title=f"{root.parent.name} / {root.name} / {scope}")
        twin.set_ylabel("Pearson r")
        handles, labels = axis.get_legend_handles_labels()
        more, more_labels = twin.get_legend_handles_labels()
        axis.legend(handles+more, labels+more_labels, fontsize=8, loc="best")
        axis.grid(alpha=.2)
        folder = root/"figures"
        folder.mkdir(exist_ok=True)
        figure.savefig(folder/f"{scope}_training_history.png", dpi=160)
        plt.close(figure)


def plot_distribution():
    frame = pd.read_csv(config.SPLITS/"distribution_summary.csv")
    figures = config.OUTPUT/"figures"
    figures.mkdir(exist_ok=True)
    figure, axes = panels(config.TARGETS)
    for axis in axes.flat[len(config.TARGETS):]:
        axis.set_visible(False)
    for axis, target in zip(axes.flat, config.TARGETS):
        subset = frame.loc[frame.target.eq(target) & frame.phase.eq("refit") & frame.split.eq("test")].sort_values("fold")
        axis.errorbar(subset.fold+1, subset.raw_median,
            yerr=[subset.raw_median-subset.raw_q10, subset.raw_q90-subset.raw_median],
            fmt="o", capsize=4, color="#0072B2")
        axis.set(title=TASK_LABELS[target], xlabel="Outer test fold", ylabel="Median, P10-P90")
        axis.set_xticks(range(1, 6))
        axis.grid(alpha=.2)
    figure.suptitle("Patient-disjoint, distribution-balanced outer test folds")
    figure.savefig(figures/"split_distribution.png", dpi=160)
    plt.close(figure)


def update(contract):
    from .run import complete
    figures, tables = config.OUTPUT/"figures", config.OUTPUT/"tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    rows, completed, fingerprints = [], {}, {}
    for name in config.VARIANTS:
        for target in config.TARGETS:
            paths = [config.OUTPUT/name/target/f"fold{fold}" for fold in range(5)]
            ready = [root for root in paths if complete(root, contract, verify=False)]
            for root in ready:
                row = pd.read_csv(root/"metrics.csv").query("split == 'test'").iloc[0].to_dict()
                rows.append(row)
                if not (root/"figures/refit_training_history.png").exists():
                    plot_history(root)
            if len(ready) == 5:
                completed[name, target] = paths
                fingerprints[f"{name}/{target}"] = [sha256(path/"video_predictions.csv") for path in paths]
    if not rows:
        return
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(tables/"fold_test_metrics.csv", index=False)
    # Rebuild full comparisons only when another target acquires all five folds.
    marker = tables/"plot_contract.json"
    if marker.exists() and json.loads(marker.read_text()) == fingerprints:
        return
    pooled, pooled_rows, means = {}, [], []
    for (name, target), _ in completed.items():
        data = pooled_frame(name, target)
        folder = config.OUTPUT/name/"oof"
        folder.mkdir(parents=True, exist_ok=True)
        data.to_csv(folder/f"{target}.csv", index=False)
        pooled[name, target] = data
        pooled_rows.append({"variant": name, "target": target, "videos": len(data),
            "patients": data.hospital_id.nunique(), **summarize(data.y_true, data.y_pred)})
        selected = fold_metrics.loc[fold_metrics.variant.eq(name) & fold_metrics.target.eq(target)]
        means.append({"variant": name, "target": target, "fold_count": 5,
            **{f"{key}_mean": selected[key].mean() for key in METRICS},
            **{f"{key}_sd": selected[key].std(ddof=1) for key in METRICS}})
    if not means:
        return
    pd.DataFrame(pooled_rows).to_csv(tables/"pooled_oof_metrics.csv", index=False)
    summary = pd.DataFrame(means)
    summary.to_csv(tables/"fold_mean_sd.csv", index=False)
    ci = []
    ci_folder = tables/"bootstrap_cache"
    ci_folder.mkdir(exist_ok=True)
    for (name, target), candidate in pooled.items():
        if name == "b00_reference" or ("b00_reference", target) not in pooled:
            continue
        baseline = pooled["b00_reference", target]
        if not np.array_equal(baseline.fold, candidate.fold):
            raise RuntimeError("Different OOF assignments across ablations")
        key = {"reference": fingerprints[f"b00_reference/{target}"], "candidate": fingerprints[f"{name}/{target}"]}
        cache = ci_folder/f"{name}_{target}.json"
        if cache.exists() and json.loads(cache.read_text())["hashes"] == key:
            result = json.loads(cache.read_text())["result"]
        else:
            from .splits import target_seed
            result = cluster_bootstrap(baseline, candidate, target_seed(target), repeats=500)
            cache.write_text(json.dumps({"hashes": key, "result": result})+"\n")
        ci.append({"variant": name, "target": target, **result})
    pd.DataFrame(ci).to_csv(tables/"paired_oof_patient_bootstrap.csv", index=False)
    for metric in METRICS:
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            sub = summary.loc[summary.target.eq(target)]
            axis.bar(np.arange(len(sub)), sub[f"{metric}_mean"], yerr=sub[f"{metric}_sd"],
                capsize=2, color=[COLORS[list(config.VARIANTS).index(name) % len(COLORS)] for name in sub.variant])
            axis.set_xticks(np.arange(len(sub)), [name.split("_")[0].upper() for name in sub.variant], rotation=45, fontsize=7)
            axis.set(title=TASK_LABELS[target], ylabel=metric)
            axis.axhline(0, color=".5", linewidth=.5)
            axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"Five-fold {metric}: mean +/- between-fold SD (complete five-fold targets only)")
        figure.savefig(figures/f"{metric}_comparison.png", dpi=170)
        plt.close(figure)
    for name in config.VARIANTS:
        if not any(key[0] == name for key in pooled):
            continue
        figure, axes = panels(config.TARGETS)
        for axis in axes.flat[len(config.TARGETS):]:
            axis.set_visible(False)
        for axis, target in zip(axes.flat, config.TARGETS):
            if (name, target) not in pooled:
                axis.set_visible(False)
                continue
            data = pooled[name, target]
            axis.scatter(data.y_true, data.y_pred, s=9, alpha=.3, color="#0072B2", linewidths=0)
            low, high = min(data.y_true.min(), data.y_pred.min()), max(data.y_true.max(), data.y_pred.max())
            axis.plot([low, high], [low, high], "--", color=".5", lw=1)
            if data.y_true.std() > 0:
                slope, intercept = np.polyfit(data.y_true, data.y_pred, 1)
                axis.plot([low, high], np.array([low, high])*slope+intercept, color="#D55E00", lw=1.5)
            axis.set(title=TASK_LABELS[target], xlabel="True", ylabel="Predicted")
            axis.grid(alpha=.15)
        figure.suptitle(f"{name}: pooled out-of-fold predictions (each patient held out once)")
        folder = config.OUTPUT/name/"figures"
        folder.mkdir(exist_ok=True)
        figure.savefig(folder/"oof_predicted_vs_true.png", dpi=170)
        plt.close(figure)
    marker.write_text(json.dumps(fingerprints, indent=2)+"\n")
