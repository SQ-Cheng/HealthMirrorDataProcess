"""Build a deterministic, patient-disjoint multi-source annotation queue."""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.raw_windows import (
    prepare_model_input,
    raw_diagnostics,
    read_clean_ecg_file,
    read_ecg_file,
)
from exp1_sqa_pretrain.sqa.weak_labels import ECGWeakLabelGenerator

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PREDICTIONS = os.path.join(
    _PKG_DIR, "sqa_outputs", "unlabeled_evaluation", "window_predictions.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(_PKG_DIR, "sqa_annotations", "round01")
DEFAULT_DATA_ROOT = "/root/shared/HealthMirrorDataset"
WINDOW_COLUMNS = {
    "tcn_window_p_qrs", "tcn_window_p_morph",
    "resnet_window_p_qrs", "resnet_window_p_morph",
}
CATEGORY_SPLIT_COUNTS = {
    "encoder_disagreement": (13, 4, 5, 3),
    "template_disagreement": (6, 2, 1, 1),
    "qrs_high_morph_low": (6, 2, 1, 1),
    "morph_high_qrs_low": (6, 2, 1, 1),
    "consensus_high_suspicious": (6, 2, 1, 1),
    "obvious_bad": (8, 2, 2, 3),
    "clean_anchor": (10, 0, 0, 0),
    "random_control": (0, 1, 4, 0),
}
SPLITS = ("train", "val", "test_random", "test_challenge")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_mirror(value):
    match = re.search(r"mirror\d+", str(value))
    return match.group(0) if match else str(value)


def _canonical_patient(value):
    match = re.search(r"patient_\d+", str(value))
    return match.group(0) if match else str(value)


def _patient_splits(frame, seed):
    rng = np.random.default_rng(seed)
    assignment = {}
    patients = frame[["canonical_mirror", "patient_key"]].drop_duplicates()
    for _, group in patients.groupby("canonical_mirror", sort=True):
        keys = group["patient_key"].to_numpy(copy=True)
        rng.shuffle(keys)
        n_val = max(1, int(round(0.15 * len(keys))))
        n_test = max(1, int(round(0.15 * len(keys))))
        if n_val + n_test >= len(keys):
            n_val, n_test = 1, int(len(keys) >= 3)
        for key in keys[:n_test]:
            assignment[key] = "test"
        for key in keys[n_test:n_test + n_val]:
            assignment[key] = "val"
        for key in keys[n_test + n_val:]:
            assignment[key] = "train"
    return assignment


def _rank(frame, column):
    return frame[column].rank(method="average", pct=True)


def _spectral_features(signal, sampling_rate_hz):
    x = np.asarray(signal, dtype=np.float64) - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / sampling_rate_hz)
    total = float(spectrum[(frequencies >= 0.05) & (frequencies <= 50.0)].sum()) + 1e-12
    baseline = float(spectrum[(frequencies >= 0.05) & (frequencies <= 0.7)].sum()) / total
    high_frequency = float(spectrum[(frequencies >= 30.0) & (frequencies <= 50.0)].sum()) / total
    return baseline, high_frequency


def _longest_flat_fraction(ecg):
    finite = np.asarray(ecg)[np.isfinite(ecg)]
    if len(finite) < 2:
        return 1.0
    low, high = np.percentile(finite, [5.0, 95.0])
    tolerance = max(1e-10, 1e-6 * float(high - low))
    flat = np.abs(np.diff(finite)) <= tolerance
    longest = current = 0
    for value in flat:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest / max(1, len(finite) - 1))


def _enrich_raw_features(frame, window_sec, target_length):
    generator = ECGWeakLabelGenerator(template_source="window")
    rows = {}
    target_fs = target_length / window_sec
    print(f"[Features] Computing diagnostic SQIs for {len(frame)} raw windows...")
    for file_index, (path, group) in enumerate(frame.groupby("file_path", sort=False)):
        timestamps, ecg = read_ecg_file(path)
        if timestamps is None:
            continue
        for index, row in group.iterrows():
            start = int(np.searchsorted(timestamps, float(row["start_time"]), side="left"))
            end = int(np.searchsorted(timestamps, float(row["start_time"]) + window_sec, side="right"))
            segment_time, segment_ecg = timestamps[start:end], ecg[start:end]
            if len(segment_time) < 16:
                continue
            model_input = prepare_model_input(segment_time, segment_ecg, window_sec, target_length)
            weak = generator(model_input, target_fs)
            baseline, high_frequency = _spectral_features(model_input, target_fs)
            features = weak["features"]
            rows[index] = {
                "baseline_power_ratio": baseline,
                "high_frequency_power_ratio": high_frequency,
                "longest_flat_fraction": _longest_flat_fraction(segment_ecg),
                "rpeak_agreement_score": features["rpeak_agreement_score"],
                "rr_plausibility_score": features["rr_plausibility_score"],
                "template_corr_score": features["template_corr_score"],
                "bad_segment_score": features["bad_segment_score"],
                "autocorr_score": features["autocorr_score"],
            }
        if (file_index + 1) % 500 == 0:
            print(f"[Features] files={file_index + 1}")
    enriched = pd.DataFrame.from_dict(rows, orient="index")
    output = frame.copy()
    for column in enriched.columns:
        output[column] = enriched[column]
    if output[list(enriched.columns)].isna().any().any():
        raise RuntimeError("Could not compute diagnostics for every prediction row.")
    return output


def _add_model_scores(frame):
    output = frame.copy()
    probability_columns = [c for c in output if c.endswith(("_p_qrs", "_p_morph"))]
    for column in probability_columns:
        output[f"{column}_rank"] = _rank(output, column)

    output["encoder_disagreement_score"] = np.maximum(
        (output["tcn_window_p_qrs_rank"] - output["resnet_window_p_qrs_rank"]).abs(),
        (output["tcn_window_p_morph_rank"] - output["resnet_window_p_morph_rank"]).abs(),
    )
    reference_columns = {
        "tcn_reference_p_qrs_rank", "tcn_reference_p_morph_rank",
        "resnet_reference_p_qrs_rank", "resnet_reference_p_morph_rank",
    }
    if reference_columns.issubset(output.columns):
        output["template_disagreement_score"] = pd.concat([
            (output["tcn_window_p_qrs_rank"] - output["tcn_reference_p_qrs_rank"]).abs(),
            (output["tcn_window_p_morph_rank"] - output["tcn_reference_p_morph_rank"]).abs(),
            (output["resnet_window_p_qrs_rank"] - output["resnet_reference_p_qrs_rank"]).abs(),
            (output["resnet_window_p_morph_rank"] - output["resnet_reference_p_morph_rank"]).abs(),
        ], axis=1).max(axis=1)
    else:
        output["template_disagreement_score"] = 0.0

    output["window_mean_qrs"] = output[["tcn_window_p_qrs", "resnet_window_p_qrs"]].mean(axis=1)
    output["window_mean_morph"] = output[["tcn_window_p_morph", "resnet_window_p_morph"]].mean(axis=1)
    output["qrs_high_morph_low_score"] = output["window_mean_qrs"] - output["window_mean_morph"]
    output["morph_high_qrs_low_score"] = -output["qrs_high_morph_low_score"]
    output["consensus_score"] = output[[
        "tcn_window_p_qrs", "tcn_window_p_morph",
        "resnet_window_p_qrs", "resnet_window_p_morph",
    ]].min(axis=1)
    artifact_proxy = pd.concat([
        _rank(output, "artifact_burden"),
        _rank(output, "flat_fraction"),
        _rank(output, "clipping_fraction"),
        _rank(output, "impulse_ratio"),
        1.0 - _rank(output, "window_mean_qrs"),
    ], axis=1).max(axis=1)
    output["diagnostic_bad_proxy"] = artifact_proxy
    return output


def _add_diagnostic_scores(frame):
    output = frame.copy()
    baseline_rank = _rank(output, "baseline_power_ratio")
    hf_rank = _rank(output, "high_frequency_power_ratio")
    output["consensus_high_suspicious_score"] = output["consensus_score"] * np.maximum(
        baseline_rank, hf_rank
    )
    output["suspicious_subtype"] = np.where(
        baseline_rank >= hf_rank, "baseline_drift", "high_frequency_noise"
    )
    flat_score = np.maximum(
        np.clip((output["flat_fraction"] - 0.05) / 0.30, 0.0, 1.0),
        np.clip(output["longest_flat_fraction"] / 0.05, 0.0, 1.0),
    )
    clipping_score = np.clip(
        (output["clipping_fraction"] - 0.02) / 0.20, 0.0, 1.0
    )
    impulse_score = output["bad_segment_score"].clip(0.0, 1.0)
    rr_score = 1.0 - output["rr_plausibility_score"]
    output["obvious_bad_score"] = pd.concat([
        pd.Series(flat_score, index=output.index),
        pd.Series(clipping_score, index=output.index),
        pd.Series(impulse_score, index=output.index),
        rr_score, output["bad_segment_score"],
    ], axis=1).max(axis=1)
    subtype_scores = pd.DataFrame({
        "flatline_or_dropout": flat_score,
        "clipping": clipping_score,
        "impulse": impulse_score,
        "rr_inconsistent": rr_score,
    }, index=output.index)
    output["artifact_subtype"] = subtype_scores.idxmax(axis=1)
    return output


def _diagnostic_subset(frame, score_column, per_split):
    parts = []
    for split, count in (("train", per_split[0]), ("val", per_split[1]), ("test", per_split[2])):
        part = (
            frame[frame["patient_split"] == split]
            .sort_values(score_column, ascending=False)
            .drop_duplicates("patient_key")
            .head(count)
        )
        parts.append(part)
    return pd.concat(parts).index


def _clean_anchor_pool(data_root, split_assignment, start_window_id, window_sec, target_length, seed):
    paths = sorted(glob.glob(os.path.join(data_root, "mirror*_auto_cleaned_sqi", "patient_*.csv")))
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    generator = ECGWeakLabelGenerator(template_source="window")
    target_fs = target_length / window_sec
    records = []
    for path in paths:
        mirror_dir = os.path.basename(os.path.dirname(path))
        mirror = _canonical_mirror(mirror_dir)
        patient = _canonical_patient(os.path.basename(path))
        patient_key = f"{mirror}/{patient}"
        if split_assignment.get(patient_key) != "train":
            continue
        timestamps, ecg = read_clean_ecg_file(path)
        if timestamps is None or timestamps[-1] - timestamps[0] < window_sec:
            continue
        start_time = float(timestamps[0])
        end = int(np.searchsorted(timestamps, start_time + window_sec, side="right"))
        segment_time, segment_ecg = timestamps[:end], ecg[:end]
        if len(segment_time) < 16 or np.isfinite(segment_ecg).sum() < 16:
            continue
        model_input = prepare_model_input(segment_time, segment_ecg, window_sec, target_length)
        weak = generator(model_input, target_fs)
        baseline, high_frequency = _spectral_features(model_input, target_fs)
        diagnostics = raw_diagnostics(segment_ecg)
        features = weak["features"]
        record = {
            "window_id": start_window_id + len(records),
            "mirror": mirror_dir,
            "canonical_mirror": mirror,
            "patient_id": patient,
            "patient_key": patient_key,
            "file_path": path,
            "window_index": 0,
            "start_time": start_time,
            "source_sampling_rate_hz": float((len(segment_time) - 1) / (segment_time[-1] - segment_time[0])),
            "data_source": "clean",
            "baseline_power_ratio": baseline,
            "high_frequency_power_ratio": high_frequency,
            "longest_flat_fraction": _longest_flat_fraction(segment_ecg),
            "rpeak_agreement_score": features["rpeak_agreement_score"],
            "rr_plausibility_score": features["rr_plausibility_score"],
            "template_corr_score": features["template_corr_score"],
            "bad_segment_score": features["bad_segment_score"],
            "autocorr_score": features["autocorr_score"],
            "clean_anchor_score": float(0.4 * features["rpeak_agreement_score"] + 0.3 * features["template_corr_score"] + 0.3 * features["autocorr_score"]),
            "artifact_subtype": "none",
        }
        record.update(diagnostics)
        records.append(record)
        if len(records) >= 40:
            break
    if len(records) < 10:
        raise RuntimeError(f"Only {len(records)} usable train-patient clean anchors found.")
    return pd.DataFrame(records)


def _category_pool(frame, category):
    if category == "qrs_high_morph_low":
        mask = (frame["window_mean_qrs"] >= 0.65) & (frame["qrs_high_morph_low_score"] > 0)
    elif category == "morph_high_qrs_low":
        mask = (frame["window_mean_morph"] >= 0.65) & (frame["morph_high_qrs_low_score"] > 0)
    elif category == "consensus_high_suspicious":
        mask = frame[["tcn_window_p_qrs", "tcn_window_p_morph", "resnet_window_p_qrs", "resnet_window_p_morph"]].min(axis=1) >= 0.65
    else:
        mask = np.ones(len(frame), dtype=bool)
    return frame.loc[mask]


def _select_top(pool, category, count, patient_split, used_patients, rng):
    if count == 0:
        return pd.DataFrame()
    eligible = pool[pool["patient_split"] == patient_split]
    eligible = eligible[~eligible["patient_key"].isin(used_patients)]
    if category == "random_control":
        eligible = eligible.sample(frac=1.0, random_state=int(rng.integers(2**31 - 1)))
        selected = eligible.drop_duplicates("patient_key").head(count).copy()
    elif category in {"obvious_bad", "consensus_high_suspicious"}:
        eligible = _category_pool(eligible, category).sort_values(
            f"{category}_score", ascending=False
        )
        if category == "obvious_bad":
            subtype_column = "artifact_subtype"
            subtype_order = (
                "flatline_or_dropout", "clipping", "impulse", "rr_inconsistent"
            )
        else:
            subtype_column = "suspicious_subtype"
            subtype_order = ("baseline_drift", "high_frequency_noise")
        subtype_rows = {
            subtype: group.to_dict("records")
            for subtype, group in eligible.groupby(subtype_column)
        }
        positions = {subtype: 0 for subtype in subtype_order}
        chosen, chosen_patients = [], set()
        while len(chosen) < count:
            progressed = False
            for subtype in subtype_order:
                rows = subtype_rows.get(subtype, [])
                while positions[subtype] < len(rows):
                    row = rows[positions[subtype]]
                    positions[subtype] += 1
                    if row["patient_key"] in chosen_patients:
                        continue
                    chosen.append(row)
                    chosen_patients.add(row["patient_key"])
                    progressed = True
                    break
                if len(chosen) >= count:
                    break
            if not progressed:
                break
        if len(chosen) < count:
            for row in eligible.to_dict("records"):
                if row["patient_key"] not in chosen_patients:
                    chosen.append(row)
                    chosen_patients.add(row["patient_key"])
                if len(chosen) >= count:
                    break
        selected = pd.DataFrame(chosen)
    else:
        score_column = f"{category}_score"
        eligible = _category_pool(eligible, category).sort_values(score_column, ascending=False)
        selected = eligible.drop_duplicates("patient_key").head(count).copy()
    if len(selected) != count:
        raise RuntimeError(
            f"Could only select {len(selected)}/{count} {category} samples from {patient_split}."
        )
    used_patients.update(selected["patient_key"])
    selected["acquisition_type"] = category
    selected["acquisition_score"] = (
        np.nan if category == "random_control" else selected[f"{category}_score"]
    )
    selected["selection_reason"] = category
    return selected


def _write_atomic_csv(frame, path):
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_queue(
    predictions_path,
    output_dir,
    data_root=DEFAULT_DATA_ROOT,
    count=100,
    repeat_count=5,
    seed=42,
    window_sec=10.0,
    target_length=1024,
):
    expected_unique = sum(sum(values) for values in CATEGORY_SPLIT_COUNTS.values())
    if count != expected_unique + repeat_count or repeat_count != 5:
        raise ValueError(f"Round 01 design requires count={expected_unique + 5}, repeat_count=5.")
    frame = pd.read_csv(predictions_path)
    missing = WINDOW_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required window-model predictions: {sorted(missing)}")

    frame["canonical_mirror"] = frame["mirror"].map(_canonical_mirror)
    frame["patient_id"] = frame["patient_id"].map(_canonical_patient)
    frame["patient_key"] = frame["canonical_mirror"] + "/" + frame["patient_id"]
    frame["data_source"] = "raw"
    split_assignment = _patient_splits(frame, seed)
    frame["patient_split"] = frame["patient_key"].map(split_assignment)
    frame = _add_model_scores(frame)

    suspicious_base = frame[frame["consensus_score"] >= 0.65]
    suspicious_indices = _diagnostic_subset(
        suspicious_base, "consensus_score", (80, 40, 40)
    )
    bad_indices = _diagnostic_subset(
        frame, "diagnostic_bad_proxy", (50, 25, 25)
    )
    for proxy_column in (
        "flat_fraction", "clipping_fraction", "impulse_ratio", "artifact_burden"
    ):
        bad_indices = bad_indices.union(
            _diagnostic_subset(frame, proxy_column, (20, 10, 10))
        )
    diagnostic_indices = suspicious_indices.union(bad_indices)
    diagnostic = _enrich_raw_features(
        frame.loc[diagnostic_indices], window_sec, target_length
    )
    diagnostic = _add_diagnostic_scores(diagnostic)
    diagnostic_columns = [
        "baseline_power_ratio", "high_frequency_power_ratio",
        "longest_flat_fraction", "rpeak_agreement_score",
        "rr_plausibility_score", "template_corr_score",
        "bad_segment_score", "autocorr_score",
        "consensus_high_suspicious_score", "obvious_bad_score",
        "artifact_subtype", "suspicious_subtype",
    ]
    for column in diagnostic_columns:
        frame[column] = diagnostic[column]

    clean_pool = _clean_anchor_pool(
        data_root, split_assignment, int(frame["window_id"].max()) + 1,
        window_sec, target_length, seed,
    )
    clean_pool["patient_split"] = "train"

    rng = np.random.default_rng(seed)
    used_patients = set()
    selected = []
    patient_split_for_output = {
        "train": "train", "val": "val",
        "test_random": "test", "test_challenge": "test",
    }
    for category, counts in CATEGORY_SPLIT_COUNTS.items():
        for split, category_count in zip(SPLITS, counts):
            if category_count == 0:
                continue
            if category == "clean_anchor":
                eligible = clean_pool[~clean_pool["patient_key"].isin(used_patients)]
                part = eligible.sort_values("clean_anchor_score", ascending=False).drop_duplicates("patient_key").head(category_count).copy()
                if len(part) != category_count:
                    raise RuntimeError("Insufficient non-overlapping clean anchors.")
                used_patients.update(part["patient_key"])
                part["acquisition_type"] = category
                part["acquisition_score"] = part["clean_anchor_score"]
                part["selection_reason"] = "high-quality auto-cleaned anchor"
            else:
                part = _select_top(
                    frame, category, category_count,
                    patient_split_for_output[split], used_patients, rng,
                )
            part["split"] = split
            selected.append(part)

    unique = pd.concat(selected, ignore_index=True, sort=False)
    unique["behavior_group"] = unique["acquisition_type"]
    suspicious = unique["acquisition_type"] == "consensus_high_suspicious"
    obvious_bad = unique["acquisition_type"] == "obvious_bad"
    unique.loc[suspicious, "behavior_group"] = unique.loc[suspicious, "suspicious_subtype"]
    unique.loc[obvious_bad, "behavior_group"] = unique.loc[obvious_bad, "artifact_subtype"]
    raw_missing = unique[
        (unique["data_source"] == "raw") & unique["baseline_power_ratio"].isna()
    ]
    if len(raw_missing):
        final_diagnostics = _add_diagnostic_scores(
            _enrich_raw_features(raw_missing, window_sec, target_length)
        )
        for column in diagnostic_columns:
            unique.loc[raw_missing.index, column] = final_diagnostics[column].to_numpy()
    if len(unique) != count - repeat_count or unique["patient_key"].duplicated().any():
        raise RuntimeError("Unique selection count or patient isolation invariant failed.")
    unique.insert(0, "queue_id", [f"r01_{index:04d}" for index in range(len(unique))])
    unique["is_repeat"] = False
    unique["repeat_of_queue_id"] = ""

    repeat_source = unique.sample(n=repeat_count, random_state=int(rng.integers(2**31 - 1))).copy()
    repeat_source["repeat_of_queue_id"] = repeat_source["queue_id"]
    repeat_source["queue_id"] = [f"r01_{index:04d}" for index in range(len(unique), count)]
    repeat_source["is_repeat"] = True
    repeat_source["acquisition_type"] = "blind_repeat"
    repeat_source["selection_reason"] = "blinded intra-rater repeat"

    queue = pd.concat([unique, repeat_source], ignore_index=True, sort=False)
    permutation = rng.permutation(len(queue))
    queue["display_order"] = np.empty(len(queue), dtype=int)
    queue.loc[permutation, "display_order"] = np.arange(len(queue))
    queue = queue.sort_values("queue_id").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    queue_path = os.path.join(output_dir, "queue.csv")
    if os.path.exists(queue_path):
        raise FileExistsError(f"Queue already exists and is immutable: {queue_path}")
    _write_atomic_csv(queue, queue_path)

    split_payload = {
        split: sorted(key for key, assigned in split_assignment.items() if assigned == split)
        for split in ("train", "val", "test")
    }
    with open(os.path.join(output_dir, "split_v1.json"), "w", encoding="utf-8") as file:
        json.dump(split_payload, file, indent=2)
    config = {
        "version": "round01_multisource_v2",
        "seed": seed,
        "task_count": count,
        "unique_window_count": count - repeat_count,
        "repeat_count": repeat_count,
        "category_split_counts": {key: dict(zip(SPLITS, values)) for key, values in CATEGORY_SPLIT_COUNTS.items()},
        "counts": queue["acquisition_type"].value_counts().to_dict(),
        "splits": queue["split"].value_counts().to_dict(),
        "predictions_path": os.path.abspath(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "data_root": os.path.abspath(data_root),
        "label_scheme": "overall bad/uncertain/good with optional task overrides",
    }
    with open(os.path.join(output_dir, "annotation_config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print(f"Saved {len(queue)} annotation tasks to {queue_path}")
    print(queue["split"].value_counts().to_string())
    print(queue["acquisition_type"].value_counts().to_string())
    return queue


def parse_args():
    parser = argparse.ArgumentParser(description="Select a multi-source ECG SQA annotation batch")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    build_queue(
        args.predictions, args.output_dir, data_root=args.data_root,
        count=args.count, repeat_count=args.repeat_count, seed=args.seed,
        window_sec=args.window_sec, target_length=args.target_length,
    )


if __name__ == "__main__":
    main()
