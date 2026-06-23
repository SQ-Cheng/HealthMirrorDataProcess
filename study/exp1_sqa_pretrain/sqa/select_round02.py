"""Build round-02 ECG SQA annotation queue from current model failures."""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.model import load_sqa_checkpoint
from exp1_sqa_pretrain.sqa.raw_windows import (
    CLEAN_ECG_POLARITY,
    RAW_ECG_POLARITY,
    extract_window_from_arrays,
    load_raw_windows,
    read_clean_ecg_file,
    read_ecg_file,
)
from exp1_sqa_pretrain.sqa.select_for_annotation import (
    _canonical_mirror,
    _canonical_patient,
    _patient_splits,
)
from exp1_sqa_pretrain.sqa.weak_labels import ECGWeakLabelGenerator

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = "/root/shared/HealthMirrorDataset"
DEFAULT_OUTPUT_DIR = os.path.join(_PKG_DIR, "sqa_annotations", "round02")
DEFAULT_ROUND01_QUEUE = os.path.join(
    _PKG_DIR, "sqa_annotations", "round01", "queue.csv"
)
DEFAULT_CHECKPOINTS = {
    "tcn_window": os.path.join(
        _PKG_DIR, "checkpoints",
        "exp1_sqa_ecg_tcn_window_L1024_run01_best.pt",
    ),
    "resnet_window": os.path.join(
        _PKG_DIR, "checkpoints",
        "exp1_sqa_ecg_resnet_window_L1024_run03_best.pt",
    ),
    "tcn_reference": os.path.join(
        _PKG_DIR, "checkpoints",
        "exp1_sqa_ecg_tcn_reference_L1024_run02_best.pt",
    ),
    "resnet_reference": os.path.join(
        _PKG_DIR, "checkpoints",
        "exp1_sqa_ecg_resnet_reference_L1024_run04_best.pt",
    ),
}
SPLITS = ("train", "val", "test_random", "test_challenge")
# Counts are (train, val, test_random, test_challenge).
SUBCATEGORY_COUNTS = {
    "clean_anchor": (16, 4, 3, 2),
    "raw_good": (16, 4, 3, 2),
    "natural_high_frequency": (10, 2, 2, 1),
    "gaussian_augmented": (6, 2, 1, 1),
    "high_frequency_augmented": (6, 2, 1, 1),
    "natural_clipping": (3, 1, 1, 0),
    "clipping_augmented": (19, 5, 3, 3),
    "natural_baseline": (10, 2, 2, 1),
    "baseline_augmented": (6, 2, 1, 1),
    "natural_impulse_dropout": (10, 2, 2, 1),
    "impulse_augmented": (3, 1, 1, 0),
    "dropout_augmented": (3, 1, 1, 0),
    "encoder_disagreement": (22, 5, 4, 4),
    "template_disagreement": (13, 3, 2, 2),
    "random_stratified": (26, 6, 5, 3),
}
CATEGORY_MAP = {
    "clean_anchor": "positive_anchor",
    "raw_good": "positive_anchor",
    "natural_high_frequency": "gaussian_high_frequency",
    "gaussian_augmented": "gaussian_high_frequency",
    "high_frequency_augmented": "gaussian_high_frequency",
    "natural_clipping": "clipping_saturation",
    "clipping_augmented": "clipping_saturation",
    "natural_baseline": "baseline_drift",
    "baseline_augmented": "baseline_drift",
    "natural_impulse_dropout": "impulse_dropout",
    "impulse_augmented": "impulse_dropout",
    "dropout_augmented": "impulse_dropout",
    "encoder_disagreement": "encoder_disagreement",
    "template_disagreement": "template_disagreement",
    "random_stratified": "random_stratified",
}
EXPECTED_CATEGORY_COUNTS = {
    "positive_anchor": 50,
    "gaussian_high_frequency": 35,
    "clipping_saturation": 35,
    "baseline_drift": 25,
    "impulse_dropout": 25,
    "encoder_disagreement": 35,
    "template_disagreement": 20,
    "random_stratified": 40,
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic_csv(frame, path):
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _rank(series):
    return series.rank(method="average", pct=True)


def _spectral_ratios(inputs, sampling_rate_hz):
    values = inputs[:, 0, :].astype(np.float64)
    values -= values.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(values, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / sampling_rate_hz)
    total = spectrum[:, (frequencies >= 0.05) & (frequencies <= 50.0)].sum(axis=1) + 1e-12
    baseline = spectrum[:, (frequencies >= 0.05) & (frequencies <= 0.7)].sum(axis=1) / total
    high_frequency = spectrum[:, (frequencies >= 30.0) & (frequencies <= 50.0)].sum(axis=1) / total
    return baseline, high_frequency


@torch.no_grad()
def _predict(model, inputs, batch_size, device):
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        outputs.append(model.predict_proba(batch).cpu().numpy())
    return np.concatenate(outputs)


def _load_models(checkpoints, device):
    models = {}
    metadata = {}
    for name, path in checkpoints.items():
        model, checkpoint = load_sqa_checkpoint(
            path, map_location="cpu", freeze_encoder=True
        )
        if (
            checkpoint.get("task") == "ecg_sqa_human"
            or "human_evaluation" in checkpoint
            or "source_sqa_checkpoint" in checkpoint
        ):
            raise ValueError(
                f"Round-02 selection requires a pre-human-label model, got: {path}"
            )
        models[name] = model.to(device).eval()
        metadata[name] = checkpoint
    return models, metadata


def _attach_predictions(frame, inputs, models, batch_size, device, names=None):
    output = frame.copy()
    names = list(models) if names is None else names
    for name in names:
        probabilities = _predict(models[name], inputs, batch_size, device)
        output[f"{name}_p_qrs"] = probabilities[:, 0]
        output[f"{name}_p_morph"] = probabilities[:, 1]
    return output


def _add_raw_scores(frame, inputs, target_fs):
    output = frame.copy()
    baseline, high_frequency = _spectral_ratios(inputs, target_fs)
    output["baseline_power_ratio"] = baseline
    output["high_frequency_power_ratio"] = high_frequency
    output["window_mean_score"] = output[[
        "tcn_window_p_qrs", "tcn_window_p_morph",
        "resnet_window_p_qrs", "resnet_window_p_morph",
    ]].mean(axis=1)
    output["window_min_score"] = output[[
        "tcn_window_p_qrs", "tcn_window_p_morph",
        "resnet_window_p_qrs", "resnet_window_p_morph",
    ]].min(axis=1)
    output["encoder_disagreement_score"] = pd.concat([
        (output["tcn_window_p_qrs"] - output["resnet_window_p_qrs"]).abs(),
        (output["tcn_window_p_morph"] - output["resnet_window_p_morph"]).abs(),
    ], axis=1).max(axis=1)
    output["template_disagreement_score"] = pd.concat([
        (output["tcn_window_p_qrs"] - output["tcn_reference_p_qrs"]).abs(),
        (output["tcn_window_p_morph"] - output["tcn_reference_p_morph"]).abs(),
        (output["resnet_window_p_qrs"] - output["resnet_reference_p_qrs"]).abs(),
        (output["resnet_window_p_morph"] - output["resnet_reference_p_morph"]).abs(),
    ], axis=1).max(axis=1)

    artifact_rank = pd.concat([
        _rank(output["artifact_burden"]),
        _rank(output["flat_fraction"]),
        _rank(output["clipping_fraction"]),
        _rank(output["impulse_ratio"]),
    ], axis=1).max(axis=1)
    spectral_artifact = np.maximum(
        _rank(output["baseline_power_ratio"]),
        _rank(output["high_frequency_power_ratio"]),
    )
    output["raw_good_score"] = (
        output["window_min_score"]
        * (1.0 - 0.60 * artifact_rank)
        * (1.0 - 0.35 * spectral_artifact)
    )
    output["natural_high_frequency_score"] = (
        output["window_mean_score"] * _rank(output["high_frequency_power_ratio"])
    )
    output["natural_clipping_score"] = (
        output["window_mean_score"] * _rank(output["clipping_fraction"])
    )
    output["natural_baseline_score"] = (
        output["window_mean_score"] * _rank(output["baseline_power_ratio"])
    )
    output["natural_impulse_dropout_score"] = (
        output["window_mean_score"] * artifact_rank
    )
    output["score_stratum"] = pd.qcut(
        output["window_mean_score"], q=4, labels=False, duplicates="drop"
    ).astype(int)
    return output


def _exclude_round01(frame, queue_path):
    if not os.path.exists(queue_path):
        return frame
    old = pd.read_csv(queue_path)
    old = old[~old.get("is_repeat", False).astype(bool)]
    old_keys = set(zip(old["file_path"], old["start_time"].round(6)))
    keep = [
        (path, round(float(start), 6)) not in old_keys
        for path, start in zip(frame["file_path"], frame["start_time"])
    ]
    return frame.loc[keep].copy()


def _clean_pool(
    data_root, split_assignment, models, batch_size, device,
    window_sec, target_length, seed, max_candidates=500,
):
    paths = sorted(glob.glob(
        os.path.join(data_root, "mirror*_auto_cleaned_sqi", "patient_*.csv")
    ))
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    generator = ECGWeakLabelGenerator(template_source="window")
    target_fs = target_length / window_sec
    records, inputs, seen_patients = [], [], set()
    for path in paths:
        mirror_dir = os.path.basename(os.path.dirname(path))
        mirror = _canonical_mirror(mirror_dir)
        patient = _canonical_patient(os.path.basename(path))
        patient_key = f"{mirror}/{patient}"
        split = split_assignment.get(patient_key)
        if split is None or patient_key in seen_patients:
            continue
        timestamps, ecg = read_clean_ecg_file(path)
        if timestamps is None or timestamps[-1] - timestamps[0] < window_sec:
            continue
        try:
            window = extract_window_from_arrays(
                timestamps, ecg, float(timestamps[0]),
                window_sec=window_sec, target_length=target_length,
                source_name=path, polarity=CLEAN_ECG_POLARITY,
            )
        except ValueError:
            continue
        weak = generator(window["model_input"], target_fs)
        features = weak["features"]
        quality = float(
            0.30 * features["rpeak_agreement_score"]
            + 0.25 * features["rr_plausibility_score"]
            + 0.25 * features["template_corr_score"]
            + 0.20 * features["autocorr_score"]
        ) * (1.0 - features["bad_segment_score"])
        record = {
            "mirror": mirror_dir,
            "canonical_mirror": mirror,
            "patient_id": patient,
            "patient_key": patient_key,
            "patient_split": split,
            "file_path": path,
            "window_index": 0,
            "start_time": float(window["timestamps"][0]),
            "source_sampling_rate_hz": window["source_sampling_rate_hz"],
            "data_source": "clean",
            "polarity": CLEAN_ECG_POLARITY,
            "corruption_type": "none",
            "corruption_severity": 0,
            "corruption_seed": 0,
            "weak_quality_score": quality,
            "base_window_id": -1,
            "sample_kind": "clean",
        }
        record.update(window["diagnostics"])
        records.append(record)
        inputs.append(window["model_input"])
        seen_patients.add(patient_key)
        if len(records) >= max_candidates:
            break
    if not records:
        raise RuntimeError("No clean anchor candidates were found.")
    frame = pd.DataFrame(records)
    values = np.stack(inputs)[:, None, :].astype(np.float32)
    frame = _attach_predictions(
        frame, values, models, batch_size, device,
        names=["tcn_window", "resnet_window"],
    )
    model_mean = frame[[
        "tcn_window_p_qrs", "tcn_window_p_morph",
        "resnet_window_p_qrs", "resnet_window_p_morph",
    ]].mean(axis=1)
    frame["clean_anchor_score"] = frame["weak_quality_score"] * (
        1.0 + 0.35 * (1.0 - model_mean)
    )
    return frame


def _augmentation_base(frame, per_split=(300, 100, 100)):
    parts = []
    for split, count in zip(("train", "val", "test"), per_split):
        parts.append(
            frame[frame["patient_split"] == split]
            .sort_values("raw_good_score", ascending=False)
            .drop_duplicates("patient_key")
            .head(count)
        )
    return pd.concat(parts, ignore_index=True)


def _augmented_pools(
    base, corruptions, models, batch_size, device,
    window_sec, target_length, seed,
):
    candidates = {}
    materialized = {corruption: {} for corruption in corruptions}
    for corruption_type in corruptions:
        candidate = base.copy().reset_index(drop=True)
        candidate["base_window_id"] = candidate["window_id"].astype(int)
        candidate["sample_kind"] = "augmented"
        candidate["corruption_type"] = corruption_type
        candidate["corruption_severity"] = [
            1 + (index % 2) for index in range(len(candidate))
        ]
        candidate["corruption_seed"] = [
            int(seed + 100003 * index + 997 * sum(map(ord, corruption_type)))
            for index in range(len(candidate))
        ]
        candidates[corruption_type] = candidate

    # Each long raw file is read once, then all requested corruptions are derived.
    for path, group in base.reset_index(drop=True).groupby("file_path", sort=False):
        timestamps, ecg = read_ecg_file(path)
        if timestamps is None:
            continue
        for index, row in group.iterrows():
            for corruption_type in corruptions:
                candidate_row = candidates[corruption_type].iloc[index]
                try:
                    window = extract_window_from_arrays(
                        timestamps, ecg, row["start_time"],
                        window_sec=window_sec, target_length=target_length,
                        source_name=path, polarity=RAW_ECG_POLARITY,
                        corruption_type=corruption_type,
                        corruption_severity=candidate_row["corruption_severity"],
                        corruption_seed=candidate_row["corruption_seed"],
                    )
                except ValueError:
                    continue
                materialized[corruption_type][index] = window["model_input"]

    outputs = {}
    for corruption_type in corruptions:
        indices = sorted(materialized[corruption_type])
        candidate = candidates[corruption_type].loc[indices].reset_index(drop=True)
        inputs = np.stack([
            materialized[corruption_type][index] for index in indices
        ])[:, None, :].astype(np.float32)
        candidate = _attach_predictions(
            candidate, inputs, models, batch_size, device,
            names=["tcn_window", "resnet_window"],
        )
        # Reference models are not needed for controlled-corruption selection;
        # do not retain the uncorrupted base-window reference probabilities.
        for architecture in ("tcn", "resnet"):
            for task in ("qrs", "morph"):
                candidate[f"{architecture}_reference_p_{task}"] = np.nan
        after = candidate[[
            "tcn_window_p_qrs", "tcn_window_p_morph",
            "resnet_window_p_qrs", "resnet_window_p_morph",
        ]].mean(axis=1)
        before = candidate["window_mean_score"]
        candidate[f"{corruption_type}_augmented_score"] = (
            after + 0.50 * np.maximum(after - before, 0.0)
        )
        outputs[corruption_type] = candidate
    return outputs


def _select_top(pool, score_column, count, patient_split, used_patients):
    eligible = pool[
        (pool["patient_split"] == patient_split)
        & ~pool["patient_key"].isin(used_patients)
    ].sort_values(score_column, ascending=False)
    selected = eligible.drop_duplicates("patient_key").head(count).copy()
    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)}/{count} candidates for {score_column} in {patient_split}."
        )
    used_patients.update(selected["patient_key"])
    selected["acquisition_score"] = selected[score_column]
    return selected


def _select_stratified(pool, count, patient_split, used_patients, rng):
    eligible = pool[
        (pool["patient_split"] == patient_split)
        & ~pool["patient_key"].isin(used_patients)
    ]
    groups = {
        stratum: group.sample(
            frac=1.0, random_state=int(rng.integers(2**31 - 1))
        ).to_dict("records")
        for stratum, group in eligible.groupby("score_stratum")
    }
    strata = sorted(groups)
    positions = {stratum: 0 for stratum in strata}
    selected, selected_patients = [], set()
    while len(selected) < count:
        progressed = False
        for stratum in strata:
            rows = groups[stratum]
            while positions[stratum] < len(rows):
                row = rows[positions[stratum]]
                positions[stratum] += 1
                if row["patient_key"] in selected_patients:
                    continue
                selected.append(row)
                selected_patients.add(row["patient_key"])
                progressed = True
                break
            if len(selected) >= count:
                break
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Only {len(selected)}/{count} stratified candidates in {patient_split}."
        )
    output = pd.DataFrame(selected)
    used_patients.update(output["patient_key"])
    output["acquisition_score"] = np.nan
    return output


def build_queue(args):
    checkpoints = {
        "tcn_window": args.tcn_window_checkpoint,
        "resnet_window": args.resnet_window_checkpoint,
        "tcn_reference": args.tcn_reference_checkpoint,
        "resnet_reference": args.resnet_reference_checkpoint,
    }
    for path in checkpoints.values():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    output_dir = os.path.abspath(args.output_dir)
    queue_path = os.path.join(output_dir, "queue.csv")
    if os.path.exists(queue_path):
        raise FileExistsError(f"Queue already exists and is immutable: {queue_path}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    models, metadata = _load_models(checkpoints, device)
    for checkpoint in metadata.values():
        if int(checkpoint["target_length"]) != args.target_length:
            raise ValueError("Checkpoint target length mismatch.")

    raw_inputs, raw = load_raw_windows(
        args.data_root, args.window_sec, args.target_length,
        args.windows_per_file, args.max_files,
    )
    raw["canonical_mirror"] = raw["mirror"].map(_canonical_mirror)
    raw["patient_id"] = raw["patient_id"].map(_canonical_patient)
    raw["patient_key"] = raw["canonical_mirror"] + "/" + raw["patient_id"]
    split_assignment = _patient_splits(raw, args.seed)
    raw["patient_split"] = raw["patient_key"].map(split_assignment)
    raw["data_source"] = "raw"
    raw["polarity"] = RAW_ECG_POLARITY
    raw["corruption_type"] = "none"
    raw["corruption_severity"] = 0
    raw["corruption_seed"] = 0
    raw["base_window_id"] = raw["window_id"].astype(int)
    raw["sample_kind"] = "natural"

    # load_raw_windows enforces raw polarity=-1 for every public raw-data path.
    raw = _attach_predictions(raw, raw_inputs, models, args.batch_size, device)
    raw = _add_raw_scores(raw, raw_inputs, args.target_length / args.window_sec)
    raw = _exclude_round01(raw, args.round01_queue)

    clean = _clean_pool(
        args.data_root, split_assignment, models, args.batch_size, device,
        args.window_sec, args.target_length, args.seed,
    )
    base = _augmentation_base(raw)
    augmented = _augmented_pools(
        base,
        (
            "gaussian", "high_frequency", "clipping",
            "baseline", "impulse", "dropout",
        ),
        models, args.batch_size, device,
        args.window_sec, args.target_length, args.seed,
    )

    pools = {
        "clean_anchor": (clean, "clean_anchor_score"),
        "raw_good": (raw, "raw_good_score"),
        "natural_high_frequency": (raw, "natural_high_frequency_score"),
        "gaussian_augmented": (augmented["gaussian"], "gaussian_augmented_score"),
        "high_frequency_augmented": (
            augmented["high_frequency"], "high_frequency_augmented_score"
        ),
        "natural_clipping": (raw, "natural_clipping_score"),
        "clipping_augmented": (augmented["clipping"], "clipping_augmented_score"),
        "natural_baseline": (raw, "natural_baseline_score"),
        "baseline_augmented": (augmented["baseline"], "baseline_augmented_score"),
        "natural_impulse_dropout": (raw, "natural_impulse_dropout_score"),
        "impulse_augmented": (augmented["impulse"], "impulse_augmented_score"),
        "dropout_augmented": (augmented["dropout"], "dropout_augmented_score"),
        "encoder_disagreement": (raw, "encoder_disagreement_score"),
        "template_disagreement": (raw, "template_disagreement_score"),
        "random_stratified": (raw, None),
    }
    patient_split_for_output = {
        "train": "train", "val": "val",
        "test_random": "test", "test_challenge": "test",
    }
    rng = np.random.default_rng(args.seed)
    used_patients, selected_parts = set(), []
    for subcategory, counts in SUBCATEGORY_COUNTS.items():
        pool, score_column = pools[subcategory]
        for split, count in zip(SPLITS, counts):
            if count == 0:
                continue
            patient_split = patient_split_for_output[split]
            if subcategory == "random_stratified":
                part = _select_stratified(
                    pool, count, patient_split, used_patients, rng
                )
            else:
                part = _select_top(
                    pool, score_column, count, patient_split, used_patients
                )
            part["split"] = split
            part["selection_subcategory"] = subcategory
            part["acquisition_type"] = CATEGORY_MAP[subcategory]
            selected_parts.append(part)

    unique = pd.concat(selected_parts, ignore_index=True, sort=False)
    expected_unique = sum(sum(counts) for counts in SUBCATEGORY_COUNTS.values())
    if len(unique) != expected_unique or unique["patient_key"].duplicated().any():
        raise RuntimeError("Round-02 count or unique-patient invariant failed.")
    category_counts = unique["acquisition_type"].value_counts().to_dict()
    if category_counts != EXPECTED_CATEGORY_COUNTS:
        raise RuntimeError(f"Category count mismatch: {category_counts}")
    if not (unique.loc[unique["data_source"] == "raw", "polarity"] == RAW_ECG_POLARITY).all():
        raise RuntimeError("Every raw round-02 sample must use polarity=-1.")
    if not (unique.loc[unique["data_source"] == "clean", "polarity"] == CLEAN_ECG_POLARITY).all():
        raise RuntimeError("Every clean round-02 sample must use polarity=+1.")

    next_window_id = int(raw_inputs.shape[0])
    non_natural = unique["sample_kind"] != "natural"
    unique.loc[non_natural, "window_id"] = np.arange(
        next_window_id, next_window_id + int(non_natural.sum())
    )
    unique["window_id"] = unique["window_id"].astype(int)
    unique.insert(0, "queue_id", [f"r02_{index:04d}" for index in range(len(unique))])
    unique["is_repeat"] = False
    unique["repeat_of_queue_id"] = ""

    repeat_source = unique.sample(
        n=args.repeat_count, random_state=int(rng.integers(2**31 - 1))
    ).copy()
    repeat_source["repeat_of_queue_id"] = repeat_source["queue_id"]
    repeat_source["queue_id"] = [
        f"r02_{index:04d}"
        for index in range(len(unique), len(unique) + args.repeat_count)
    ]
    repeat_source["is_repeat"] = True
    repeat_source["selection_subcategory"] = "blind_repeat"
    repeat_source["acquisition_type"] = "blind_repeat"

    queue = pd.concat([unique, repeat_source], ignore_index=True, sort=False)
    permutation = rng.permutation(len(queue))
    queue["display_order"] = np.empty(len(queue), dtype=int)
    queue.loc[permutation, "display_order"] = np.arange(len(queue))
    queue = queue.sort_values("queue_id").reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    _write_atomic_csv(queue, queue_path)
    split_payload = {
        split: sorted(key for key, value in split_assignment.items() if value == split)
        for split in ("train", "val", "test")
    }
    with open(os.path.join(output_dir, "split_v2.json"), "w", encoding="utf-8") as file:
        json.dump(split_payload, file, indent=2)
    config = {
        "version": "round02_targeted_failures_v1",
        "seed": args.seed,
        "task_count": len(queue),
        "unique_window_count": len(unique),
        "repeat_count": args.repeat_count,
        "category_counts": category_counts,
        "subcategory_counts": unique["selection_subcategory"].value_counts().to_dict(),
        "split_counts_unique": unique["split"].value_counts().to_dict(),
        "raw_polarity": -1.0,
        "clean_polarity": 1.0,
        "selection_model_constraint": "all four checkpoints predate human labels",
        "checkpoints": {
            name: {"path": os.path.abspath(path), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
        "data_root": os.path.abspath(args.data_root),
        "round01_queue": os.path.abspath(args.round01_queue),
        "controlled_corruptions": {
            "types": [
                "gaussian", "high_frequency", "clipping",
                "baseline", "impulse", "dropout",
            ],
            "severity_levels_used": [1, 2],
            "application_order": "polarity then native-scale corruption then zscore",
        },
    }
    with open(
        os.path.join(output_dir, "annotation_config.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    print(f"Saved {len(queue)} tasks ({len(unique)} unique) to {queue_path}")
    print(unique["acquisition_type"].value_counts().to_string())
    print(unique["split"].value_counts().to_string())
    print(unique.groupby(["data_source", "polarity"]).size().to_string())
    return queue


def parse_args():
    parser = argparse.ArgumentParser(description="Build targeted SQA annotation round 02")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round01-queue", default=DEFAULT_ROUND01_QUEUE)
    parser.add_argument("--tcn-window-checkpoint", default=DEFAULT_CHECKPOINTS["tcn_window"])
    parser.add_argument("--resnet-window-checkpoint", default=DEFAULT_CHECKPOINTS["resnet_window"])
    parser.add_argument("--tcn-reference-checkpoint", default=DEFAULT_CHECKPOINTS["tcn_reference"])
    parser.add_argument("--resnet-reference-checkpoint", default=DEFAULT_CHECKPOINTS["resnet_reference"])
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--windows-per-file", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--repeat-count", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    build_queue(parse_args())


if __name__ == "__main__":
    main()
