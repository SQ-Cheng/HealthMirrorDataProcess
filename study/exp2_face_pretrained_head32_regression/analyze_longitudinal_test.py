"""Analyze within-patient test-set changes using trained 20-frame regressors."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = EXP_DIR / "outputs" / "20frame"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "longitudinal_test"
ARCHITECTURES = ("mobilenet_v3_small", "efficientnet_b0")
ARCHITECTURE_LABELS = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
}
ARCHITECTURE_COLORS = {
    "mobilenet_v3_small": "#2878B5",
    "efficientnet_b0": "#D95F02",
}
TARGETS = {
    "hemoglobin_low": {
        "label": "Hemoglobin",
        "prefix": "hemoglobin",
        "unit": "g/L",
        "direction": "low",
    },
    "po2_low": {
        "label": "PO2",
        "prefix": "po2",
        "unit": "mmHg",
        "direction": "low",
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _score_to_raw(predicted_score, threshold, scale, direction):
    displacement = scale * np.sinh(predicted_score)
    if direction == "low":
        return threshold - displacement
    if direction == "high":
        return threshold + displacement
    raise ValueError(f"Unsupported score direction: {direction}")


def _load_unique_events(input_dir):
    source_path = input_dir / "source_data" / "base_manifest.csv"
    source = pd.read_csv(
        source_path,
        dtype={"hospital_id": str, "video_id": str},
    )
    all_events = []
    input_paths = [source_path]
    for target, definition in TARGETS.items():
        prefix = definition["prefix"]
        lab_time = f"{prefix}_lab_time_unix"
        lab_value = f"{prefix}_value"
        lab_delta = f"{prefix}_delta_h"
        signed_delta = f"{prefix}_signed_delta_h"
        records_path = input_dir / "task_records" / f"{target}.csv"
        records = pd.read_csv(
            records_path,
            dtype={"hospital_id": str, "video_id": str},
        )
        records = records.loc[records["split"].eq("test")].copy()
        columns = [
            "hospital_id",
            "video_id",
            "capture_time_unix",
            "capture_start_unix",
            "capture_end_unix",
            lab_time,
            lab_value,
            lab_delta,
            signed_delta,
        ]
        merged = records.merge(
            source[columns],
            on=["hospital_id", "video_id"],
            how="left",
            validate="one_to_one",
        )
        required = [
            "capture_time_unix",
            lab_time,
            lab_value,
            lab_delta,
            signed_delta,
        ]
        if merged[required].isna().any().any():
            raise ValueError(f"Missing longitudinal metadata for {target}")
        if not np.allclose(
            merged["raw_value"].to_numpy(np.float64),
            merged[lab_value].to_numpy(np.float64),
        ):
            raise ValueError(f"Task/source raw values disagree for {target}")
        if (merged[lab_delta] > 24.0 + 1e-6).any():
            raise ValueError(f"Out-of-window lab match found for {target}")
        if not np.allclose(
            merged[lab_delta],
            np.abs(merged[signed_delta]),
            atol=1e-7,
        ):
            raise ValueError(f"Signed match deltas disagree for {target}")

        merged["target"] = target
        merged["lab_time_unix"] = merged[lab_time].astype(np.float64)
        merged["lab_value"] = merged[lab_value].astype(np.float64)
        merged["lab_delta_h"] = merged[lab_delta].astype(np.float64)
        merged["capture_lab_midpoint_delta_h"] = (
            merged["capture_time_unix"] - merged["lab_time_unix"]
        ).abs() / 3600.0
        merged = merged.sort_values(
            [
                "hospital_id",
                "lab_time_unix",
                "lab_delta_h",
                "capture_lab_midpoint_delta_h",
                "video_id",
            ]
        )
        merged["videos_assigned_to_lab_event"] = merged.groupby(
            ["hospital_id", "lab_time_unix"]
        )["video_id"].transform("size")
        merged = merged.drop_duplicates(
            ["hospital_id", "lab_time_unix"], keep="first"
        )
        counts = merged.groupby("hospital_id")["lab_time_unix"].transform("size")
        merged = merged.loc[counts.ge(2)].copy()
        merged = merged.sort_values(
            ["hospital_id", "lab_time_unix", "capture_time_unix"]
        )
        if not merged.groupby("hospital_id")["capture_time_unix"].apply(
            lambda values: np.all(np.diff(values.to_numpy(np.float64)) > 0)
        ).all():
            raise ValueError(f"Capture and lab order disagree for {target}")
        all_events.append(merged)
        input_paths.append(records_path)
    return pd.concat(all_events, ignore_index=True), input_paths


def _attach_predictions(events, input_dir):
    frames = []
    input_paths = []
    for architecture in ARCHITECTURES:
        for target, definition in TARGETS.items():
            path = (
                input_dir
                / "runs"
                / architecture
                / target
                / "video_predictions.csv"
            )
            predictions = pd.read_csv(
                path,
                dtype={"hospital_id": str, "video_id": str},
            )
            predictions = predictions.loc[predictions["split"].eq("test")].copy()
            selected = events.loc[events["target"].eq(target)].merge(
                predictions[
                    [
                        "hospital_id",
                        "video_id",
                        "y_true",
                        "y_pred",
                        "frame_count",
                        "architecture",
                    ]
                ],
                on=["hospital_id", "video_id"],
                how="left",
                validate="one_to_one",
            )
            if selected["y_pred"].isna().any():
                raise ValueError(f"Missing predictions for {architecture}/{target}")
            if not selected["architecture"].eq(architecture).all():
                raise ValueError(f"Architecture mismatch in {path}")
            if not selected["frame_count"].eq(20).all():
                raise ValueError(f"Non-20-frame prediction in {path}")
            if not np.allclose(
                selected["y_true"],
                selected["abnormal_score"],
                atol=1e-5,
            ):
                raise ValueError(f"True scores disagree for {architecture}/{target}")
            selected["predicted_raw_value"] = _score_to_raw(
                selected["y_pred"].to_numpy(np.float64),
                selected["score_threshold"].to_numpy(np.float64),
                selected["score_scale"].to_numpy(np.float64),
                definition["direction"],
            )
            selected["event_index"] = selected.groupby("hospital_id").cumcount()
            selected["elapsed_lab_h"] = selected.groupby("hospital_id")[
                "lab_time_unix"
            ].transform(lambda values: (values - values.min()) / 3600.0)
            frames.append(selected)
            input_paths.append(path)
    return pd.concat(frames, ignore_index=True), input_paths


def _build_transitions(events):
    rows = []
    group_columns = ["architecture", "target", "hospital_id"]
    for (architecture, target, hospital_id), group in events.groupby(group_columns):
        group = group.sort_values("lab_time_unix").reset_index(drop=True)
        for index in range(len(group) - 1):
            first, second = group.iloc[index], group.iloc[index + 1]
            delta_raw = float(second["lab_value"] - first["lab_value"])
            delta_predicted_raw = float(
                second["predicted_raw_value"] - first["predicted_raw_value"]
            )
            delta_score = float(second["y_true"] - first["y_true"])
            delta_predicted_score = float(second["y_pred"] - first["y_pred"])
            true_sign = int(np.sign(delta_raw))
            predicted_sign = int(np.sign(delta_predicted_raw))
            rows.append(
                {
                    "architecture": architecture,
                    "target": target,
                    "hospital_id": hospital_id,
                    "transition_index": index,
                    "start_video_id": first["video_id"],
                    "end_video_id": second["video_id"],
                    "start_lab_time_unix": first["lab_time_unix"],
                    "end_lab_time_unix": second["lab_time_unix"],
                    "lab_interval_h": (
                        second["lab_time_unix"] - first["lab_time_unix"]
                    )
                    / 3600.0,
                    "start_capture_time_unix": first["capture_time_unix"],
                    "end_capture_time_unix": second["capture_time_unix"],
                    "start_lab_value": first["lab_value"],
                    "end_lab_value": second["lab_value"],
                    "delta_lab_value": delta_raw,
                    "start_predicted_raw_value": first["predicted_raw_value"],
                    "end_predicted_raw_value": second["predicted_raw_value"],
                    "delta_predicted_raw_value": delta_predicted_raw,
                    "start_true_score": first["y_true"],
                    "end_true_score": second["y_true"],
                    "delta_true_score": delta_score,
                    "start_predicted_score": first["y_pred"],
                    "end_predicted_score": second["y_pred"],
                    "delta_predicted_score": delta_predicted_score,
                    "true_change_sign": true_sign,
                    "predicted_change_sign": predicted_sign,
                    "true_value_changed": true_sign != 0,
                    "direction_concordant": (
                        true_sign != 0 and true_sign == predicted_sign
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["transition_id"] = (
        result["target"].astype(str)
        + ":"
        + result["hospital_id"].astype(str)
        + ":"
        + result["start_lab_time_unix"].astype("int64").astype(str)
        + ":"
        + result["end_lab_time_unix"].astype("int64").astype(str)
    )
    return result


def _safe_correlation(x, y, method):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    value = pearsonr(x, y).statistic if method == "pearson" else spearmanr(x, y).statistic
    return float(value)


def _metric_values(transitions):
    changed = transitions.loc[transitions["true_value_changed"]]
    if changed.empty:
        return {
            "direction_concordance": np.nan,
            "patient_macro_concordance": np.nan,
            "delta_pearson_r": np.nan,
            "delta_spearman_r": np.nan,
            "delta_mae": np.nan,
            "delta_rmse": np.nan,
            "delta_bias": np.nan,
            "delta_slope": np.nan,
        }
    true_delta = changed["delta_lab_value"].to_numpy(np.float64)
    predicted_delta = changed["delta_predicted_raw_value"].to_numpy(np.float64)
    errors = predicted_delta - true_delta
    return {
        "direction_concordance": float(changed["direction_concordant"].mean()),
        "patient_macro_concordance": float(
            changed.groupby("hospital_id")["direction_concordant"].mean().mean()
        ),
        "delta_pearson_r": _safe_correlation(
            true_delta,
            predicted_delta,
            "pearson",
        ),
        "delta_spearman_r": _safe_correlation(
            true_delta,
            predicted_delta,
            "spearman",
        ),
        "delta_mae": float(np.mean(np.abs(errors))),
        "delta_rmse": float(np.sqrt(np.mean(errors**2))),
        "delta_bias": float(np.mean(errors)),
        "delta_slope": float(np.polyfit(true_delta, predicted_delta, 1)[0]),
    }


def _weighted_pearson(x, y, weights):
    weights = np.asarray(weights, dtype=np.float64)
    keep = weights > 0
    x = np.asarray(x, dtype=np.float64)[keep]
    y = np.asarray(y, dtype=np.float64)[keep]
    weights = weights[keep]
    if len(x) < 3:
        return np.nan
    x_centered = x - np.average(x, weights=weights)
    y_centered = y - np.average(y, weights=weights)
    covariance = np.sum(weights * x_centered * y_centered)
    denominator = np.sqrt(
        np.sum(weights * x_centered**2)
        * np.sum(weights * y_centered**2)
    )
    return float(covariance / denominator) if denominator > 0 else np.nan


def _weighted_slope(x, y, weights):
    weights = np.asarray(weights, dtype=np.float64)
    keep = weights > 0
    x = np.asarray(x, dtype=np.float64)[keep]
    y = np.asarray(y, dtype=np.float64)[keep]
    weights = weights[keep]
    x_centered = x - np.average(x, weights=weights)
    denominator = np.sum(weights * x_centered**2)
    if denominator <= 0:
        return np.nan
    y_centered = y - np.average(y, weights=weights)
    return float(np.sum(weights * x_centered * y_centered) / denominator)


def _weighted_ranks(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    ranks = np.empty(len(values), dtype=np.float64)
    cumulative = 0
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        group_weight = int(sorted_weights[start:end].sum())
        rank = cumulative + (group_weight + 1) / 2.0
        ranks[order[start:end]] = rank
        cumulative += group_weight
        start = end
    return ranks


def _cluster_bootstrap(transitions, seed, repetitions):
    changed = transitions.loc[transitions["true_value_changed"]].copy()
    patient_ids = changed["hospital_id"].drop_duplicates().to_numpy()
    patient_lookup = {
        patient_id: index for index, patient_id in enumerate(patient_ids)
    }
    patient_index = changed["hospital_id"].map(patient_lookup).to_numpy(np.int32)
    true_delta = changed["delta_lab_value"].to_numpy(np.float64)
    predicted_delta = changed["delta_predicted_raw_value"].to_numpy(np.float64)
    concordant = changed["direction_concordant"].to_numpy(np.float64)
    patient_correct = np.bincount(
        patient_index,
        weights=concordant,
        minlength=len(patient_ids),
    )
    patient_transitions = np.bincount(
        patient_index,
        minlength=len(patient_ids),
    )
    patient_accuracy = patient_correct / patient_transitions
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
        patient_weights = np.bincount(sampled, minlength=len(patient_ids))
        transition_weights = patient_weights[patient_index]
        direction = float(
            np.average(concordant, weights=transition_weights)
        )
        macro = float(
            np.average(patient_accuracy, weights=patient_weights)
        )
        pearson = _weighted_pearson(
            true_delta,
            predicted_delta,
            transition_weights,
        )
        true_ranks = _weighted_ranks(true_delta, transition_weights)
        predicted_ranks = _weighted_ranks(predicted_delta, transition_weights)
        spearman = _weighted_pearson(
            true_ranks,
            predicted_ranks,
            transition_weights,
        )
        errors = predicted_delta - true_delta
        values.append(
            {
                "direction_concordance": direction,
                "patient_macro_concordance": macro,
                "delta_pearson_r": pearson,
                "delta_spearman_r": spearman,
                "delta_mae": float(
                    np.average(np.abs(errors), weights=transition_weights)
                ),
                "delta_rmse": float(
                    np.sqrt(np.average(errors**2, weights=transition_weights))
                ),
                "delta_bias": float(
                    np.average(errors, weights=transition_weights)
                ),
                "delta_slope": _weighted_slope(
                    true_delta,
                    predicted_delta,
                    transition_weights,
                ),
            }
        )
    return pd.DataFrame(values)


def _cluster_permutation_p(transitions, observed, seed, repetitions):
    changed = transitions.loc[transitions["true_value_changed"]].copy()
    patient_ids = changed["hospital_id"].drop_duplicates().to_numpy()
    patient_lookup = {
        patient_id: index for index, patient_id in enumerate(patient_ids)
    }
    patient_index = changed["hospital_id"].map(patient_lookup).to_numpy(np.int32)
    true_sign = changed["true_change_sign"].to_numpy(np.int8)
    predicted_sign = changed["predicted_change_sign"].to_numpy(np.int8)
    rng = np.random.default_rng(seed)
    extreme = 0
    observed_distance = abs(observed - 0.5)
    for _ in range(repetitions):
        patient_sign = rng.choice(
            np.asarray((-1, 1), dtype=np.int8),
            size=len(patient_ids),
        )
        statistic = float(
            np.mean(predicted_sign * patient_sign[patient_index] == true_sign)
        )
        extreme += abs(statistic - 0.5) >= observed_distance - 1e-12
    return float((extreme + 1) / (repetitions + 1))


def _summarize(transitions, seed, bootstrap_repetitions, permutation_repetitions):
    rows = []
    patient_rows = []
    for (architecture, target), group in transitions.groupby(
        ["architecture", "target"], sort=False
    ):
        changed = group.loc[group["true_value_changed"]]
        metrics = _metric_values(group)
        bootstrap = _cluster_bootstrap(
            group,
            seed + len(rows) * 1009,
            bootstrap_repetitions,
        )
        row = {
            "architecture": architecture,
            "target": target,
            "patients": int(group["hospital_id"].nunique()),
            "events": int(len(group) + group["hospital_id"].nunique()),
            "transitions": int(len(group)),
            "nonzero_true_transitions": int(len(changed)),
            "concordant_transitions": int(changed["direction_concordant"].sum()),
            **metrics,
            "naive_exact_binomial_p_vs_0_5": float(
                binomtest(
                    int(changed["direction_concordant"].sum()),
                    len(changed),
                    0.5,
                ).pvalue
            ),
            "patient_cluster_permutation_p_vs_0_5": _cluster_permutation_p(
                group,
                metrics["direction_concordance"],
                seed + len(rows) * 2027,
                permutation_repetitions,
            ),
        }
        for metric in (
            "direction_concordance",
            "patient_macro_concordance",
            "delta_pearson_r",
            "delta_spearman_r",
            "delta_mae",
            "delta_bias",
            "delta_slope",
        ):
            finite = bootstrap[metric].dropna()
            row[f"{metric}_ci_low"] = float(finite.quantile(0.025))
            row[f"{metric}_ci_high"] = float(finite.quantile(0.975))
        rows.append(row)

        for hospital_id, patient in group.groupby("hospital_id"):
            patient_changed = patient.loc[patient["true_value_changed"]]
            patient_rows.append(
                {
                    "architecture": architecture,
                    "target": target,
                    "hospital_id": hospital_id,
                    "events": int(len(patient) + 1),
                    "transitions": int(len(patient)),
                    "nonzero_true_transitions": int(len(patient_changed)),
                    "concordant_transitions": int(
                        patient_changed["direction_concordant"].sum()
                    ),
                    "direction_concordance": (
                        float(patient_changed["direction_concordant"].mean())
                        if len(patient_changed)
                        else np.nan
                    ),
                    "lab_span_h": float(
                        patient["end_lab_time_unix"].max()
                        - patient["start_lab_time_unix"].min()
                    )
                    / 3600.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(patient_rows)


def _compare_models(transitions, seed, repetitions):
    rows = []
    for target in TARGETS:
        selected = transitions.loc[
            transitions["target"].eq(target)
            & transitions["true_value_changed"]
        ]
        wide = selected.pivot(
            index=["hospital_id", "transition_id"],
            columns="architecture",
            values="direction_concordant",
        ).reset_index()
        if wide[list(ARCHITECTURES)].isna().any().any():
            raise ValueError(f"Models do not share transitions for {target}")
        first = wide[ARCHITECTURES[0]].astype(bool)
        second = wide[ARCHITECTURES[1]].astype(bool)
        observed = float(second.mean() - first.mean())
        patient_ids = wide["hospital_id"].drop_duplicates().to_numpy()
        patient_lookup = {
            patient_id: index for index, patient_id in enumerate(patient_ids)
        }
        patient_index = wide["hospital_id"].map(patient_lookup).to_numpy(np.int32)
        first_values = wide[ARCHITECTURES[0]].to_numpy(np.float64)
        second_values = wide[ARCHITECTURES[1]].to_numpy(np.float64)
        rng = np.random.default_rng(seed + len(rows) * 4099)
        differences = []
        for _ in range(repetitions):
            sampled = rng.integers(0, len(patient_ids), size=len(patient_ids))
            patient_weights = np.bincount(sampled, minlength=len(patient_ids))
            transition_weights = patient_weights[patient_index]
            differences.append(
                float(
                    np.average(second_values, weights=transition_weights)
                    - np.average(first_values, weights=transition_weights)
                )
            )
        rows.append(
            {
                "target": target,
                "patients": int(len(patient_ids)),
                "transitions": int(len(wide)),
                "mobilenet_direction_concordance": float(first.mean()),
                "efficientnet_direction_concordance": float(second.mean()),
                "efficientnet_minus_mobilenet": observed,
                "difference_ci_low": float(np.quantile(differences, 0.025)),
                "difference_ci_high": float(np.quantile(differences, 0.975)),
                "both_correct": int((first & second).sum()),
                "mobilenet_only_correct": int((first & ~second).sum()),
                "efficientnet_only_correct": int((~first & second).sum()),
                "both_incorrect": int((~first & ~second).sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_delta_scatter(transitions, metrics, figure_dir):
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), squeeze=False)
    for row, target in enumerate(TARGETS):
        definition = TARGETS[target]
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            selected = transitions.loc[
                transitions["target"].eq(target)
                & transitions["architecture"].eq(architecture)
                & transitions["true_value_changed"]
            ]
            concordant = selected["direction_concordant"]
            axis.scatter(
                selected.loc[concordant, "delta_lab_value"],
                selected.loc[concordant, "delta_predicted_raw_value"],
                s=25,
                alpha=0.68,
                color="#2A9D8F",
                edgecolors="none",
                label="Same direction",
            )
            axis.scatter(
                selected.loc[~concordant, "delta_lab_value"],
                selected.loc[~concordant, "delta_predicted_raw_value"],
                s=25,
                alpha=0.68,
                color="#D1495B",
                edgecolors="none",
                label="Opposite/no change",
            )
            axis.axhline(0, color="#555555", linewidth=0.8)
            axis.axvline(0, color="#555555", linewidth=0.8)
            limit = max(
                np.abs(selected["delta_lab_value"]).max(),
                np.abs(selected["delta_predicted_raw_value"]).max(),
                1.0,
            )
            axis.set_xlim(-limit * 1.05, limit * 1.05)
            axis.set_ylim(-limit * 1.05, limit * 1.05)
            axis.plot(
                [-limit, limit],
                [-limit, limit],
                color="#333333",
                linestyle=":",
                linewidth=1,
                label="Ideal magnitude",
            )
            if target == "po2_low":
                axis.set_xscale("symlog", linthresh=10)
                axis.set_yscale("symlog", linthresh=10)
            else:
                axis.set_aspect("equal", adjustable="box")
            row_metrics = metrics.loc[
                metrics["target"].eq(target)
                & metrics["architecture"].eq(architecture)
            ].iloc[0]
            axis.set_title(
                f"{definition['label']} | {ARCHITECTURE_LABELS[architecture]}\n"
                f"n={len(selected)}, agreement="
                f"{row_metrics['direction_concordance']:.1%}, "
                f"rho={row_metrics['delta_spearman_r']:.2f}"
            )
            axis.set_xlabel(f"True change ({definition['unit']})")
            axis.set_ylabel(f"Predicted change ({definition['unit']})")
            axis.grid(alpha=0.18)
            axis.legend(loc="best")
    figure.suptitle(
        "Within-patient adjacent changes on the held-out test set",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "delta_true_vs_predicted.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_direction_summary(metrics, figure_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), squeeze=False)
    x = np.arange(len(ARCHITECTURES))
    labels = [ARCHITECTURE_LABELS[item] for item in ARCHITECTURES]
    for axis, target in zip(axes.flat, TARGETS):
        selected = metrics.set_index(["target", "architecture"]).loc[
            [(target, architecture) for architecture in ARCHITECTURES]
        ]
        values = selected["direction_concordance"].to_numpy()
        errors = np.vstack(
            [
                values - selected["direction_concordance_ci_low"].to_numpy(),
                selected["direction_concordance_ci_high"].to_numpy() - values,
            ]
        )
        axis.bar(
            x,
            values,
            yerr=errors,
            capsize=5,
            color=[ARCHITECTURE_COLORS[item] for item in ARCHITECTURES],
            width=0.62,
        )
        axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
        for position, row in zip(x, selected.itertuples()):
            axis.text(
                position,
                min(row.direction_concordance + 0.04, 0.97),
                f"{row.concordant_transitions}/{row.nonzero_true_transitions}",
                ha="center",
                fontsize=9,
            )
        axis.set_ylim(0, 1.03)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.set_ylabel("Direction concordance")
        axis.set_title(
            f"{TARGETS[target]['label']} | patient-cluster bootstrap 95% CI"
        )
        axis.grid(axis="y", alpha=0.22)
    figure.suptitle(
        "Does the predicted within-patient change follow the true lab change?",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "direction_concordance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_patient_heatmap(transitions, figure_dir):
    figure, axes = plt.subplots(2, 2, figsize=(14, 15), squeeze=False)
    cmap = ListedColormap(["#D1495B", "#D9D9D9", "#2A9D8F"])
    for row, target in enumerate(TARGETS):
        for column, architecture in enumerate(ARCHITECTURES):
            axis = axes[row, column]
            selected = transitions.loc[
                transitions["target"].eq(target)
                & transitions["architecture"].eq(architecture)
            ].copy()
            ordering = (
                selected.groupby("hospital_id")
                .size()
                .sort_values(ascending=False, kind="stable")
            )
            patients = ordering.index.tolist()
            width = int(selected["transition_index"].max()) + 1
            matrix = np.full((len(patients), width), np.nan)
            for patient_index, hospital_id in enumerate(patients):
                patient = selected.loc[selected["hospital_id"].eq(hospital_id)]
                for transition in patient.itertuples():
                    value = (
                        0
                        if not transition.true_value_changed
                        else (1 if transition.direction_concordant else -1)
                    )
                    matrix[patient_index, transition.transition_index] = value
            masked = np.ma.masked_invalid(matrix)
            axis.imshow(
                masked,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=-1,
                vmax=1,
            )
            axis.set_xticks(np.arange(width), np.arange(1, width + 1))
            axis.set_xlabel("Adjacent transition number")
            axis.set_ylabel("Patients, sorted by number of transitions")
            axis.set_title(
                f"{TARGETS[target]['label']} | "
                f"{ARCHITECTURE_LABELS[architecture]} | n={len(patients)} patients"
            )
    figure.suptitle(
        "Patient-level direction agreement: green=same, red=opposite, gray=true tie",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "patient_direction_heatmap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_trajectory_examples(events, figure_dir):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for row, target in enumerate(TARGETS):
        target_events = events.loc[events["target"].eq(target)]
        counts = (
            target_events.loc[
                target_events["architecture"].eq(ARCHITECTURES[0])
            ]
            .groupby("hospital_id")
            .size()
            .sort_values(ascending=False, kind="stable")
        )
        selected_patients = sorted(
            counts.loc[counts.eq(counts.max())].index
        )[:3]
        if len(selected_patients) < 3:
            selected_patients = counts.index[:3].tolist()
        for column, hospital_id in enumerate(selected_patients):
            axis = axes[row, column]
            reference = target_events.loc[
                target_events["hospital_id"].eq(hospital_id)
                & target_events["architecture"].eq(ARCHITECTURES[0])
            ].sort_values("lab_time_unix")
            axis.plot(
                reference["elapsed_lab_h"] / 24.0,
                reference["lab_value"],
                color="#202020",
                marker="o",
                linewidth=1.8,
                label="True lab",
            )
            for architecture in ARCHITECTURES:
                selected = target_events.loc[
                    target_events["hospital_id"].eq(hospital_id)
                    & target_events["architecture"].eq(architecture)
                ].sort_values("lab_time_unix")
                axis.plot(
                    selected["elapsed_lab_h"] / 24.0,
                    selected["predicted_raw_value"],
                    color=ARCHITECTURE_COLORS[architecture],
                    marker="s",
                    linestyle="--",
                    linewidth=1.3,
                    label=ARCHITECTURE_LABELS[architecture],
                )
            axis.set_title(
                f"{TARGETS[target]['label']} | patient {hospital_id} | "
                f"{len(reference)} time points"
            )
            axis.set_xlabel("Days since first matched lab")
            axis.set_ylabel(
                f"{TARGETS[target]['label']} ({TARGETS[target]['unit']})"
            )
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle(
        "Pre-specified examples: patients with the most independent time points",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        figure_dir / "longitudinal_trajectory_examples.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_report(metrics, comparison, events, transitions, output_dir):
    lines = [
        "# Test-Set Longitudinal Change Analysis",
        "",
        "This analysis reuses video-level predictions from the trained 20-frame "
        "single-task regression models. One closest video is retained for each "
        "patient/lab timestamp, and only patients with at least two independent "
        "lab timestamps are included.",
        "",
        "## Sample",
        "",
        "| Target | Patients | Unique lab-video pairs | Adjacent transitions | "
        "Nonzero transitions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    first_architecture = ARCHITECTURES[0]
    for target in TARGETS:
        selected_events = events.loc[
            events["target"].eq(target)
            & events["architecture"].eq(first_architecture)
        ]
        selected_transitions = transitions.loc[
            transitions["target"].eq(target)
            & transitions["architecture"].eq(first_architecture)
        ]
        lines.append(
            f"| {TARGETS[target]['label']} | "
            f"{selected_events['hospital_id'].nunique()} | {len(selected_events)} | "
            f"{len(selected_transitions)} | "
            f"{int(selected_transitions['true_value_changed'].sum())} |"
        )
    lines.extend(
        [
            "",
            "## Direction Results",
            "",
            "| Target | Model | Concordant / nonzero | Concordance (95% patient-cluster CI) "
            "| Delta Spearman rho (95% CI) | Cluster permutation p vs 0.5 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for target in TARGETS:
        for architecture in ARCHITECTURES:
            row = metrics.loc[
                metrics["target"].eq(target)
                & metrics["architecture"].eq(architecture)
            ].iloc[0]
            lines.append(
                f"| {TARGETS[target]['label']} | "
                f"{ARCHITECTURE_LABELS[architecture]} | "
                f"{int(row['concordant_transitions'])}/"
                f"{int(row['nonzero_true_transitions'])} | "
                f"{row['direction_concordance']:.3f} "
                f"({row['direction_concordance_ci_low']:.3f}, "
                f"{row['direction_concordance_ci_high']:.3f}) | "
                f"{row['delta_spearman_r']:.3f} "
                f"({row['delta_spearman_r_ci_low']:.3f}, "
                f"{row['delta_spearman_r_ci_high']:.3f}) | "
                f"{row['patient_cluster_permutation_p_vs_0_5']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Magnitude Results",
            "",
            "| Target | Model | Delta MAE | Delta bias | Delta slope "
            "(95% patient-cluster CI) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for target in TARGETS:
        for architecture in ARCHITECTURES:
            row = metrics.loc[
                metrics["target"].eq(target)
                & metrics["architecture"].eq(architecture)
            ].iloc[0]
            unit = TARGETS[target]["unit"]
            lines.append(
                f"| {TARGETS[target]['label']} | "
                f"{ARCHITECTURE_LABELS[architecture]} | "
                f"{row['delta_mae']:.2f} {unit} | "
                f"{row['delta_bias']:+.2f} {unit} | "
                f"{row['delta_slope']:.3f} "
                f"({row['delta_slope_ci_low']:.3f}, "
                f"{row['delta_slope_ci_high']:.3f}) |"
            )
    lines.extend(
        [
            "",
            "## Model Comparison",
            "",
            "| Target | EfficientNet minus MobileNet concordance | 95% patient-cluster CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {TARGETS[row.target]['label']} | "
            f"{row.efficientnet_minus_mobilenet:+.3f} | "
            f"({row.difference_ci_low:+.3f}, {row.difference_ci_high:+.3f}) |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Direction concordance is calculated only where the true lab value changed.",
            "- Confidence intervals resample patients, preserving all transitions within "
            "each sampled patient.",
            "- The exact-binomial p-value in the machine table treats transitions as "
            "independent and is secondary; the patient-cluster permutation result is "
            "the preferred significance check.",
            "- This is a selected held-out longitudinal subset, not a new external test set.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(input_dir, output_dir, seed, bootstrap_repetitions, permutation_repetitions):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    _style()

    unique_events, metadata_paths = _load_unique_events(input_dir)
    events, prediction_paths = _attach_predictions(unique_events, input_dir)
    transitions = _build_transitions(events)
    metrics, patient_metrics = _summarize(
        transitions,
        seed,
        bootstrap_repetitions,
        permutation_repetitions,
    )
    comparison = _compare_models(transitions, seed, bootstrap_repetitions)

    events.to_csv(table_dir / "longitudinal_events.csv", index=False)
    transitions.to_csv(table_dir / "longitudinal_transitions.csv", index=False)
    metrics.to_csv(table_dir / "longitudinal_metrics.csv", index=False)
    patient_metrics.to_csv(
        table_dir / "longitudinal_patient_metrics.csv", index=False
    )
    comparison.to_csv(table_dir / "model_comparison.csv", index=False)

    _plot_delta_scatter(transitions, metrics, figure_dir)
    _plot_direction_summary(metrics, figure_dir)
    _plot_patient_heatmap(transitions, figure_dir)
    _plot_trajectory_examples(events, figure_dir)
    _write_report(metrics, comparison, events, transitions, output_dir)

    with open(output_dir / "analysis_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "analysis": "within_patient_longitudinal_change_concordance",
                "input_dir": str(input_dir),
                "split": "test",
                "frame_policy": "20 deterministic non-adjacent frames per video",
                "prediction_aggregation": "mean frame prediction per video",
                "targets": list(TARGETS),
                "architectures": list(ARCHITECTURES),
                "event_deduplication": (
                    "one video per patient/lab timestamp; minimum interval distance, "
                    "then minimum capture-midpoint distance, then video ID"
                ),
                "transition_policy": "adjacent distinct lab timestamps within patient",
                "direction_denominator": "transitions with nonzero true raw-value change",
                "bootstrap": {
                    "unit": "patient",
                    "repetitions": bootstrap_repetitions,
                    "seed": seed,
                },
                "permutation": {
                    "unit": "patient sign flip",
                    "repetitions": permutation_repetitions,
                    "seed": seed,
                },
                "input_sha256": {
                    str(path.relative_to(input_dir)): _sha256(path)
                    for path in (*metadata_paths, *prediction_paths)
                },
                "outputs": {
                    "human_report": "REPORT.md",
                    "figures": sorted(path.name for path in figure_dir.glob("*.png")),
                    "tables": sorted(path.name for path in table_dir.glob("*.csv")),
                },
            },
            handle,
            indent=2,
        )
    print(
        f"Saved longitudinal analysis: events={len(events)} "
        f"transitions={len(transitions)} output={output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--permutation-repetitions", type=int, default=10000)
    arguments = parser.parse_args()
    main(
        arguments.input_dir,
        arguments.output_dir,
        arguments.seed,
        arguments.bootstrap_repetitions,
        arguments.permutation_repetitions,
    )
