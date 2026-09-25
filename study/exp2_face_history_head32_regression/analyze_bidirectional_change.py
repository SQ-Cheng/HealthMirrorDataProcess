"""Test whether paired predictions follow both rising and falling lab values."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_figsize, target_grid_shape

from .config import TARGETS
from .plot_results import TASK_LABELS, TASK_UNITS


STUDY_DIR = Path(__file__).resolve().parent.parent
HISTORY_ROOT = STUDY_DIR / "exp2_face_history_head32_regression/outputs/20frame"
MODELS = {
    "face_history": (
        HISTORY_ROOT,
        "Face + history",
        "efficientnet_b0",
    ),
    "face_only": (
        STUDY_DIR / "exp2_face_pretrained_head32_regression/outputs/20frame",
        "Face only",
        "efficientnet_b0",
    ),
    "history_only": (
        STUDY_DIR / "exp2_history_only_head32_regression/outputs",
        "History only",
        None,
    ),
}
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_REPLICATES = 2000
COLORS = {"rise": "#2878B5", "fall": "#D05B3E"}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_events(root, architecture, target, reference):
    run_dir = root / "runs" / target
    if architecture:
        run_dir = root / "runs" / architecture / target
    predictions_path = run_dir / "video_predictions.csv"
    predictions = pd.read_csv(
        predictions_path, dtype={"hospital_id": str, "video_id": str}
    )
    predictions = predictions.loc[predictions["split"].eq("test")].copy()
    records = pd.read_csv(
        root / "task_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    records = records.loc[records["split"].eq("test")]
    if predictions["video_id"].duplicated().any() or records["video_id"].duplicated().any():
        raise AssertionError(f"Duplicate test video IDs: {root}, {target}")
    if set(predictions.video_id) != set(records.video_id):
        raise AssertionError(f"Test prediction/record video IDs differ: {root}, {target}")
    joined = predictions.merge(
        records[["video_id", "hospital_id", "raw_value"]].rename(
            columns={"hospital_id": "record_hospital_id", "raw_value": "record_raw_value"}
        ),
        on="video_id", validate="one_to_one",
    )
    if not joined.hospital_id.eq(joined.record_hospital_id).all():
        raise AssertionError(f"Patient ID mismatch: {root}, {target}")
    if not np.allclose(joined.y_true, joined.record_raw_value, rtol=0, atol=1e-9):
        raise AssertionError(f"Prediction/label raw values differ: {root}, {target}")
    joined = joined.merge(
        reference, on="video_id", validate="one_to_one",
    )
    if not joined.hospital_id.eq(joined.reference_hospital_id).all():
        raise AssertionError(f"Patient ID mismatch with reference: {root}, {target}")
    if not np.allclose(joined.y_true, joined.reference_raw_value, rtol=0, atol=1e-9):
        raise AssertionError(f"Label mismatch with reference: {root}, {target}")
    key = [
        "hospital_id", "episode_admission_time_unix", "episode_discharge_time_unix",
        "current_lab_time_unix",
    ]
    if joined.groupby(key).y_true.nunique().gt(1).any():
        raise AssertionError(f"Conflicting values at one lab event: {root}, {target}")
    events = joined.groupby(key, as_index=False).agg(
        true_value=("y_true", "first"),
        predicted_value=("y_pred", "mean"),
        video_count=("video_id", "nunique"),
        video_ids=("video_id", lambda values: "|".join(sorted(values))),
        first_video_time_unix=("capture_time_unix", "min"),
        last_video_time_unix=("capture_time_unix", "max"),
    )
    events["target"] = target
    return events, predictions_path


def _reference(target):
    records = pd.read_csv(
        HISTORY_ROOT / "task_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    records = records.loc[records["split"].eq("test")]
    summary = pd.read_csv(
        HISTORY_ROOT / "history_records" / f"{target}_summary.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    base = pd.read_csv(
        HISTORY_ROOT / "source_data/base_manifest.csv",
        usecols=["video_id", "capture_time_unix"],
        dtype={"video_id": str},
    )
    reference = records[["video_id", "hospital_id", "raw_value"]].rename(
        columns={"hospital_id": "reference_hospital_id", "raw_value": "reference_raw_value"}
    )
    reference = reference.merge(
        summary[[
            "video_id", "current_lab_time_unix", "episode_admission_time_unix",
            "episode_discharge_time_unix",
        ]], on="video_id", validate="one_to_one",
    ).merge(base, on="video_id", validate="one_to_one")
    if reference.isna().any().any():
        raise AssertionError(f"Missing test reference data for {target}")
    return reference


def _make_pairs(events):
    keys = ["hospital_id", "episode_admission_time_unix", "episode_discharge_time_unix"]
    events = events.sort_values(keys + ["current_lab_time_unix"]).copy()
    grouped = events.groupby(keys, sort=False)
    pairs = events.copy()
    for column in (
        "current_lab_time_unix", "true_value", "predicted_value", "video_ids",
        "video_count", "last_video_time_unix",
    ):
        pairs[f"previous_{column}"] = grouped[column].shift()
    pairs = pairs.loc[pairs.previous_current_lab_time_unix.notna()].copy()
    if not pairs.current_lab_time_unix.gt(pairs.previous_current_lab_time_unix).all():
        raise AssertionError("Non-increasing laboratory event time")
    if not pairs.first_video_time_unix.gt(pairs.previous_last_video_time_unix).all():
        raise AssertionError("Video time order disagrees with laboratory order")
    pairs["true_delta"] = pairs.true_value - pairs.previous_true_value
    pairs["predicted_delta"] = pairs.predicted_value - pairs.previous_predicted_value
    pairs["hours_between_labs"] = (
        pairs.current_lab_time_unix - pairs.previous_current_lab_time_unix
    ) / 3600
    pairs["true_direction"] = np.where(
        pairs.true_delta.gt(0), "rise", np.where(pairs.true_delta.lt(0), "fall", "tie")
    )
    pairs["predicted_direction"] = np.where(
        pairs.predicted_delta.gt(0), "rise",
        np.where(pairs.predicted_delta.lt(0), "fall", "tie"),
    )
    pairs["correct_direction"] = pairs.true_direction.eq(pairs.predicted_direction)
    return pairs


def _direction_metrics(pairs, rng):
    rows = []
    evaluable = pairs.loc[pairs.true_direction.ne("tie")].copy()
    patient_ids = evaluable.hospital_id.drop_duplicates().to_numpy()
    for direction in ("rise", "fall"):
        subset = evaluable.loc[evaluable.true_direction.eq(direction)]
        if subset.empty:
            rows.append((direction, 0, 0, np.nan, np.nan, np.nan))
            continue
        count = len(subset)
        correct = int(subset.correct_direction.sum())
        per_patient = subset.groupby("hospital_id").correct_direction.agg(["sum", "count"])
        patient_index = pd.Index(patient_ids)
        hits = per_patient["sum"].reindex(patient_index, fill_value=0).to_numpy(float)
        totals = per_patient["count"].reindex(patient_index, fill_value=0).to_numpy(float)
        draws = rng.integers(0, len(patient_ids), size=(BOOTSTRAP_REPLICATES, len(patient_ids)))
        numerator = hits[draws].sum(axis=1)
        denominator = totals[draws].sum(axis=1)
        rates = numerator[denominator > 0] / denominator[denominator > 0]
        rows.append((
            direction, count, int(subset.hospital_id.nunique()),
            correct / count, float(np.quantile(rates, 0.025)),
            float(np.quantile(rates, 0.975)),
        ))
    return rows


def _plot_recall(metrics, figure_dir, model_label):
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for index, direction in enumerate(("rise", "fall")):
        selected = metrics.loc[metrics.true_direction.eq(direction)].set_index("target")
        x = np.arange(len(TARGETS)) + (index - 0.5) * 0.28
        values = selected.loc[list(TARGETS), "recall"].to_numpy(float)
        low = selected.loc[list(TARGETS), "ci_low"].to_numpy(float)
        high = selected.loc[list(TARGETS), "ci_high"].to_numpy(float)
        mask = np.isfinite(values) & np.isfinite(low) & np.isfinite(high)
        axis.errorbar(
            x[mask], values[mask],
            yerr=np.vstack((values[mask] - low[mask], high[mask] - values[mask])),
            fmt="o", capsize=3, color=COLORS[direction], label=direction.title(),
        )
        for position, value, n in zip(x, values, selected.loc[list(TARGETS), "pairs"]):
            if np.isfinite(value):
                axis.annotate(f"n={n}", (position, value), xytext=(0, 8),
                              textcoords="offset points", ha="center", fontsize=7)
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axis.set(ylim=(-0.05, 1.12), ylabel="Correct predicted direction / actual direction")
    axis.set_xticks(np.arange(len(TARGETS)), [TASK_LABELS[t] for t in TARGETS], rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.suptitle(f"{model_label}: rise and fall recall on adjacent test lab events")
    figure.tight_layout()
    figure.savefig(figure_dir / "direction_recall.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_deltas(pairs, figure_dir, model_label):
    rows, columns = target_grid_shape(len(TARGETS))
    figure, axes = plt.subplots(
        rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False,
    )
    for axis, target in zip(axes.flat, TARGETS):
        selected = pairs.loc[pairs.target.eq(target) & pairs.true_direction.ne("tie")]
        values = np.concatenate((selected.true_delta.to_numpy(), selected.predicted_delta.to_numpy()))
        limit = max(float(np.quantile(np.abs(values), 0.98)), 1e-9)
        for direction in ("rise", "fall"):
            side = selected.loc[selected.true_direction.eq(direction)]
            axis.scatter(side.true_delta, side.predicted_delta, s=18, alpha=0.64,
                         color=COLORS[direction], label=f"{direction.title()} n={len(side)}")
        axis.plot((-limit, limit), (-limit, limit), color="#777777", linestyle=":")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit),
                 xlabel=f"Actual change ({TASK_UNITS[target]})",
                 ylabel=f"Predicted change ({TASK_UNITS[target]})",
                 title=TASK_LABELS[target])
        axis.grid(alpha=0.15)
        axis.legend(fontsize=7)
    figure.suptitle(f"{model_label}: adjacent test-event changes")
    figure.tight_layout()
    figure.savefig(figure_dir / "predicted_vs_actual_delta.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run():
    source_hashes = {}
    metric_rows = {model: [] for model in MODELS}
    pair_tables = {model: [] for model in MODELS}
    audit_rows = {model: [] for model in MODELS}
    for target_index, target in enumerate(TARGETS):
        reference = _reference(target)
        aligned = None
        for model, (root, _, architecture) in MODELS.items():
            events, prediction_path = _load_events(root, architecture, target, reference)
            pairs = _make_pairs(events)
            pair_tables[model].append(pairs)
            audit_rows[model].append({
                "target": target, "test_videos": len(reference),
                "unique_lab_events": len(events), "adjacent_pairs": len(pairs),
                "true_ties_excluded": int(pairs.true_direction.eq("tie").sum()),
                "evaluable_pairs": int(pairs.true_direction.ne("tie").sum()),
                "evaluable_patients": int(pairs.loc[pairs.true_direction.ne("tie"), "hospital_id"].nunique()),
            })
            key = ["hospital_id", "episode_admission_time_unix", "current_lab_time_unix"]
            identity = pairs[key + ["true_delta"]].reset_index(drop=True)
            if aligned is None:
                aligned = identity
            else:
                pd.testing.assert_frame_equal(aligned, identity, check_exact=True)
            rng = np.random.default_rng(BOOTSTRAP_SEED + target_index)
            for direction, count, patients, recall, low, high in _direction_metrics(pairs, rng):
                metric_rows[model].append({
                    "target": target, "true_direction": direction, "pairs": count,
                    "patients": patients, "recall": recall, "ci_low": low, "ci_high": high,
                    "predicted_ties": int(pairs.loc[pairs.true_direction.eq(direction), "predicted_direction"].eq("tie").sum()),
                })
            source_hashes[f"{model}/{target}"] = _sha256(prediction_path)
    for model, (root, label, _) in MODELS.items():
        output_dir = root / "bidirectional_change_analysis"
        figure_dir = output_dir / "figures"
        table_dir = output_dir / "tables"
        figure_dir.mkdir(parents=True, exist_ok=True)
        table_dir.mkdir(parents=True, exist_ok=True)
        pairs = pd.concat(pair_tables[model], ignore_index=True)
        metrics = pd.DataFrame(metric_rows[model])
        pairs.to_csv(table_dir / "adjacent_test_event_pairs.csv", index=False)
        metrics.to_csv(table_dir / "direction_recall.csv", index=False)
        pd.DataFrame(audit_rows[model]).to_csv(table_dir / "eligibility_audit.csv", index=False)
        _plot_recall(metrics, figure_dir, label)
        _plot_deltas(pairs, figure_dir, label)
        manifest = {
            "analysis": "bidirectional longitudinal change, without retraining",
            "model": model,
            "data": "saved test video_predictions.csv; raw laboratory values",
            "event": "same patient, same admission, same lab timestamp; average video predictions",
            "pair": "adjacent distinct lab timestamps with strictly ordered video capture times",
            "direction": "sign(later value - earlier value), separately for predicted and actual values",
            "ties": "exclude actual ties; predicted ties count as errors",
            "confidence_interval": "95% percentile bootstrap by patient, 2000 replicates",
            "seed": BOOTSTRAP_SEED,
            "targets": list(TARGETS),
            "prediction_sha256": {target: source_hashes[f"{model}/{target}"] for target in TARGETS},
            "reference_sha256": {
                target: {
                    "task_records": _sha256(HISTORY_ROOT / "task_records" / f"{target}.csv"),
                    "history_summary": _sha256(HISTORY_ROOT / "history_records" / f"{target}_summary.csv"),
                }
                for target in TARGETS
            },
            "base_manifest_sha256": _sha256(HISTORY_ROOT / "source_data/base_manifest.csv"),
        }
        (table_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{model}: {output_dir}")
        print(metrics[["target", "true_direction", "pairs", "recall"]].to_string(index=False))


if __name__ == "__main__":
    run()
