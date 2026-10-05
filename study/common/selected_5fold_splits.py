"""Shared patient-disjoint, distribution-balanced folds for selected Exp2/Exp3 runs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import StratifiedKFold

from study.exp2_binary_classification_common.engine import PREPARED_DIR, REFERENCE_DIR
from study.exp2_face_pretrained_head32_regression.config import (
    SCORE_DEFINITIONS, SEED, TARGETS,
)
from study.exp2_face_pretrained_head32_regression.scaling import (
    fit_robust_target_scaler,
)


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
EXP3_DIR = ROOT / "study/exp3_video_lab_regression/outputs"
SPLIT_ROOT = BASE_DIR.parent / "5fold/splits"
FOLDS = 5
CANDIDATES = 256


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(target):
    name = f"{target}.csv"
    classification = PREPARED_DIR if target == "total_bilirubin_high" else REFERENCE_DIR
    return (BASE_DIR / "task_records" / name,
            classification / "task_records" / name,
            EXP3_DIR / "task_records" / name)


def _load_canonical(target):
    paths = _source_paths(target)
    frames = [pd.read_csv(path, dtype={"hospital_id": str, "video_id": str})
              for path in paths]
    base = frames[0].sort_values("video_id").reset_index(drop=True)
    if base.video_id.duplicated().any():
        raise ValueError(f"Duplicate video: {target}")
    for other in frames[1:]:
        other = other.sort_values("video_id").reset_index(drop=True)
        stable = ["video_id", "hospital_id", "source_sample_id", "binary_label"]
        pd.testing.assert_frame_equal(
            base[stable], other[stable], check_dtype=False, check_exact=True
        )
        if not np.allclose(base.raw_value, other.raw_value, rtol=0, atol=1e-10):
            raise AssertionError(f"Raw target differs across experiments: {target}")
    if not np.isfinite(base[["raw_value", "abnormal_score"]].to_numpy(float)).all():
        raise ValueError(f"Nonfinite split-balancing values: {target}")
    return base, [_sha256(path) for path in paths]


def _score(records, row_folds, patient_folds):
    video_fraction = np.bincount(row_folds, minlength=FOLDS) / len(row_folds)
    patient_fraction = np.bincount(patient_folds, minlength=FOLDS) / len(patient_folds)
    size_error = max(np.abs(video_fraction - 0.2).max(),
                     np.abs(patient_fraction - 0.2).max())
    labels = records.binary_label.to_numpy(float)
    rates = np.array([labels[row_folds == fold].mean() for fold in range(FOLDS)])
    distribution = []
    for column in ("raw_value", "abnormal_score"):
        values = records[column].to_numpy(float)
        iqr = max(float(np.quantile(values, .75) - np.quantile(values, .25)), 1e-9)
        global_quantiles = np.quantile(values, (.1, .25, .5, .75, .9))
        for fold in range(FOLDS):
            held = values[row_folds == fold]
            distribution.append((
                wasserstein_distance(values, held) / iqr,
                float(ks_2samp(values, held).statistic),
                float(np.max(np.abs(np.quantile(held, (.1, .25, .5, .75, .9))
                                    - global_quantiles)) / iqr),
            ))
    worst = np.max(distribution, axis=0)
    objective = (2 * worst[0] + worst[1] + .25 * worst[2]
                 + .5 * (rates.max() - rates.min()) + .25 * size_error)
    return {
        "objective": float(objective), "max_wasserstein_iqr": float(worst[0]),
        "max_ks": float(worst[1]), "max_quantile_iqr": float(worst[2]),
        "positive_rate_range": float(rates.max() - rates.min()),
        "max_size_fraction_error": float(size_error),
    }


def choose_folds(records, target, candidates=CANDIDATES):
    patients = records.groupby("hospital_id", sort=True).binary_label.max()
    labels = patients.to_numpy(int)
    if len(np.unique(labels)) != 2 or min(np.bincount(labels)) < FOLDS:
        raise ValueError(f"Insufficient patient classes for five folds: {target}")
    row_patient = patients.index.get_indexer(records.hospital_id)
    offset = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")
    best = None
    for attempt in range(candidates):
        splitter = StratifiedKFold(
            n_splits=FOLDS, shuffle=True,
            random_state=(SEED + offset + attempt) % (2**32 - 1),
        )
        patient_folds = np.empty(len(patients), dtype=np.int8)
        for fold, (_, held) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            patient_folds[held] = fold
        row_folds = patient_folds[row_patient]
        if any(len(np.unique(records.binary_label.to_numpy()[row_folds == fold])) < 2
               for fold in range(FOLDS)):
            continue
        score = _score(records, row_folds, patient_folds)
        key = (score["objective"], score["max_wasserstein_iqr"], attempt)
        if best is None or key < best[0]:
            best = (key, patient_folds.copy(), score, attempt)
    if best is None:
        raise RuntimeError(f"No valid patient folds for {target}")
    _, patient_folds, score, selected = best
    return patients.index.to_numpy(str), patient_folds, score, selected


def prepare_splits(source_loader=None, split_root=None, source_policy=None):
    split_root = Path(split_root) if split_root is not None else SPLIT_ROOT
    source_loader = source_loader or _load_canonical
    sources = {target: source_loader(target) for target in TARGETS}
    hashes = {target: fingerprints for target, (_, fingerprints) in sources.items()}
    manifest_path = split_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_sha256"] != hashes:
            raise RuntimeError("Existing five-fold split source differs")
        if source_policy is not None and manifest.get("source_policy") != source_policy:
            raise RuntimeError("Existing five-fold source policy differs")
        expected = [split_root / f"{target}_fold{fold}.csv"
                    for target in TARGETS for fold in range(FOLDS)]
        if not all(path.is_file() for path in expected):
            raise RuntimeError("Existing five-fold split records are incomplete")
        print(f"[folds-reused] directory={split_root}", flush=True)
        return split_root
    if split_root.exists() and any(split_root.iterdir()):
        raise RuntimeError(f"Partial five-fold split output: {split_root}")
    split_root.mkdir(parents=True, exist_ok=True)
    assignment_rows, summary_rows, scalers, choices = [], [], {}, {}
    for target, (records, _) in sources.items():
        patient_ids, patient_folds, score, selected = choose_folds(records, target)
        assignment = dict(zip(patient_ids, patient_folds))
        assigned = records.hospital_id.map(assignment).to_numpy(int)
        if len(assignment) != records.hospital_id.nunique():
            raise AssertionError(f"Patient assignment incomplete: {target}")
        assignment_rows.extend({"target": target, "hospital_id": patient,
                                "fold": int(fold)}
                               for patient, fold in zip(patient_ids, patient_folds))
        choices[target] = {"selected_candidate": selected, **score}
        scalers[target] = {}
        for fold in range(FOLDS):
            split_names = np.where(assigned == fold, "test",
                                   np.where(assigned == (fold + 1) % FOLDS,
                                            "val", "train"))
            prepared = records.copy()
            prepared["split"] = split_names
            scaler = fit_robust_target_scaler(
                target, prepared, SCORE_DEFINITIONS[target]["unit"]
            )
            prepared["robust_scaled_raw_value"] = scaler.transform(prepared.raw_value)
            prepared.to_csv(split_root / f"{target}_fold{fold}.csv", index=False)
            scalers[target][str(fold)] = scaler.to_dict()
            for split in ("train", "val", "test"):
                subset = prepared.loc[prepared.split.eq(split)]
                if subset.binary_label.nunique() != 2:
                    raise AssertionError(f"Single-class {target}/{fold}/{split}")
                summary_rows.append({
                    "target": target, "fold": fold, "split": split,
                    "videos": len(subset), "patients": subset.hospital_id.nunique(),
                    "positive_videos": int(subset.binary_label.eq(1).sum()),
                    "positive_rate": float(subset.binary_label.mean()),
                    "raw_median": float(subset.raw_value.median()),
                    "raw_q10": float(subset.raw_value.quantile(.1)),
                    "raw_q90": float(subset.raw_value.quantile(.9)),
                })
        print(f"[folds-selected] target={target} candidate={selected} "
              f"objective={score['objective']:.4f}", flush=True)
    pd.DataFrame(assignment_rows).to_csv(split_root / "patient_folds.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(split_root / "distribution_summary.csv", index=False)
    (split_root / "scalers.json").write_text(
        json.dumps(scalers, indent=2), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1, "folds": FOLDS, "candidate_count": CANDIDATES,
        "seed": SEED, "validation_policy": "next fold cyclically",
        "train_policy": "remaining three folds",
        "stratification": "patient-level max binary label",
        "selection_objective": (
            "2*max_Wasserstein/IQR + max_KS + 0.25*max_quantile/IQR "
            "+ 0.5*positive_rate_range + 0.25*max_size_error; "
            "compared with full cohort over raw_value and abnormal_score"
        ),
        "source_sha256": hashes, "selections": choices,
    }
    if source_policy is not None:
        manifest["source_policy"] = source_policy
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return split_root
