"""Remove matching-distance bin means from saved five-fold OOF predictions."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS

from .analyze import BIN_NAMES, OUTPUT, load_joined


WINDOW_BINS = {6: BIN_NAMES[:2], 12: BIN_NAMES[:3]}


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def centered_predictions(joined):
    selected = joined.loc[joined["time_bin"].isin(WINDOW_BINS[12])].copy()
    means = selected.groupby(["target", "time_bin"], observed=True)[
        ["y_true", "y_pred"]
    ].transform("mean")
    selected["bin_mean_true"] = means["y_true"]
    selected["bin_mean_pred"] = means["y_pred"]
    selected["y_true_centered"] = selected["y_true"] - selected["bin_mean_true"]
    selected["y_pred_centered"] = selected["y_pred"] - selected["bin_mean_pred"]
    for field in ("y_true_centered", "y_pred_centered"):
        residual_means = selected.groupby(["target", "time_bin"], observed=True)[field].mean()
        if not np.allclose(residual_means, 0, rtol=0, atol=1e-10):
            raise AssertionError(f"Incorrect bin centering for {field}")
    return selected


def summarize(centered, targets):
    rows = []
    for target in targets:
        for hours, bins in WINDOW_BINS.items():
            group = centered.loc[centered["target"].eq(target)
                                 & centered["time_bin"].isin(bins)]
            if not set(group["time_bin"]) == set(bins):
                raise AssertionError(f"Missing matching-distance bin: {target}/{hours}h")
            rows.append({
                "target": target, "max_match_delta_h": hours,
                "included_bins": ",".join(bins),
                "n_videos": len(group),
                "n_patients": group["hospital_id"].nunique(),
                "uncentered_pearson_r": pearson(group["y_true"], group["y_pred"]),
                "within_bin_centered_pearson_r": pearson(
                    group["y_true_centered"], group["y_pred_centered"]
                ),
            })
    result = pd.DataFrame(rows)
    result["centered_minus_uncentered_r"] = (
        result["within_bin_centered_pearson_r"] - result["uncentered_pearson_r"]
    )
    return result


def plot(summary, targets):
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    values = summary[["uncentered_pearson_r", "within_bin_centered_pearson_r"]].to_numpy(float)
    lower, upper = np.nanmin(values), np.nanmax(values)
    padding = max(0.07, 0.15 * (upper - lower))
    limits = (max(-1, min(0, lower) - padding), min(1, max(0, upper) + padding))
    for axis, target in zip(axes.flat, targets):
        rows = summary.loc[summary["target"].eq(target)].set_index("max_match_delta_h")
        for offset, field, label, color in (
            (-0.17, "uncentered_pearson_r", "Uncentered", "#6D7B86"),
            (0.17, "within_bin_centered_pearson_r", "Within-bin centered", "#278078"),
        ):
            bars = axis.bar(np.arange(2) + offset, rows.loc[[6, 12], field],
                            width=0.32, label=label, color=color)
            for bar in bars:
                value = bar.get_height()
                if np.isfinite(value):
                    axis.annotate(f"{value:.2f}",
                                  (bar.get_x() + bar.get_width() / 2, value),
                                  xytext=(0, 4 if value >= 0 else -5),
                                  textcoords="offset points", ha="center",
                                  va="bottom" if value >= 0 else "top", fontsize=8)
        counts = rows.loc[[6, 12], "n_videos"].to_numpy(int)
        axis.set_xticks((0, 1), (f"0-6 h\nn={counts[0]}", f"0-12 h\nn={counts[1]}"))
        axis.set_title(TASK_LABELS.get(target, target), fontsize=11)
        axis.set_ylabel("Pearson r")
        axis.set_ylim(*limits)
        axis.axhline(0, linewidth=0.8, color="#444444")
        axis.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(loc="upper left", fontsize=8)
    figure.suptitle("24 h five-fold OOF | within matching-distance bins", fontsize=14)
    path = OUTPUT / "figures" / "within_bin_centered_pearson_r.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    joined, targets, source_hashes = load_joined()
    centered = centered_predictions(joined)
    summary = summarize(centered, targets)
    (OUTPUT / "figures").mkdir(parents=True, exist_ok=True)
    centered[[
        "fold", "target", "hospital_id", "video_id", "time_bin", "match_delta_h",
        "y_true", "y_pred", "bin_mean_true", "bin_mean_pred",
        "y_true_centered", "y_pred_centered",
    ]].to_csv(OUTPUT / "within_bin_centered_oof.csv", index=False)
    summary.to_csv(OUTPUT / "within_bin_centered_pearson_r.csv", index=False)
    plot(summary, targets)
    (OUTPUT / "within_bin_centering_manifest.json").write_text(json.dumps({
        "source": "24h five-fold held-out OOF predictions",
        "source_sha256": source_hashes,
        "analysis_unit": "one held-out video/lab pair per target; all five folds pooled",
        "centering_groups": ["target", "time_bin"],
        "centering": "subtract separate within-bin means from y_true and y_pred",
        "windows_hours": {str(hours): list(bins) for hours, bins in WINDOW_BINS.items()},
        "window_policy": "6h and 12h are post-hoc subsets of the same 24h OOF predictions; no retraining",
        "n_centered_video_predictions": len(centered),
    }, indent=2), encoding="utf-8")
    print(f"[within-bin-complete] rows={len(summary)} samples={len(centered)} output={OUTPUT}",
          flush=True)


if __name__ == "__main__":
    main()
