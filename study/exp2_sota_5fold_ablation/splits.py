"""Balanced patient folds, nested holdout and full-80% refit records."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import StratifiedKFold, train_test_split

from study.common.run_video_loss_12h import sha256
from study.exp2_sota_ablation.data import read_records
from . import config


def target_seed(target, fold=0):
    offset = int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")
    return (config.SEED + offset + 1009*fold) % (2**32-1)


def patient_strata(records):
    patients = records.groupby("hospital_id", sort=True).raw_value.median()
    if len(patients) < 25 or not np.isfinite(records.raw_value).all():
        raise ValueError("Five-fold analysis requires >=25 patients and finite values")
    # Patient ranks allow tied lab values while retaining near-equal stratum sizes.
    bins = min(5, len(patients)//5)
    labels = pd.qcut(patients.rank(method="first"), bins, labels=False).to_numpy(int)
    return patients.index.to_numpy(str), labels


def distribution_score(records, assignment, fractions):
    values = records.raw_value.to_numpy(float)
    iqr = max(float(np.subtract(*np.quantile(values, [.75, .25]))), 1e-9)
    global_q = np.quantile(values, [.1, .25, .5, .75, .9])
    row_groups = records.hospital_id.map(assignment).to_numpy(int)
    scores = []
    for group, fraction in enumerate(fractions):
        held = values[row_groups == group]
        if not len(held):
            return float("inf")
        scores.append(2*wasserstein_distance(values, held)/iqr
                      + ks_2samp(values, held).statistic
                      + .25*np.max(np.abs(np.quantile(held, [.1, .25, .5, .75, .9])-global_q))/iqr
                      + .5*abs(len(held)/len(values)-fraction))
    return float(max(scores))


def choose_outer(records, target, candidates=None):
    patients, strata = patient_strata(records)
    best = None
    for candidate in range(candidates or config.OUTER_CANDIDATES):
        splitter = StratifiedKFold(5, shuffle=True, random_state=(target_seed(target)+candidate) % (2**32-1))
        folds = np.empty(len(patients), dtype=int)
        for fold, (_, held) in enumerate(splitter.split(patients, strata)):
            folds[held] = fold
        assignment = dict(zip(patients, folds))
        score = distribution_score(records, assignment, [.2]*5)
        if best is None or score < best[0]:
            best = (score, assignment, candidate)
    return best


def choose_inner(outer_train, target, fold, candidates=None):
    patients, strata = patient_strata(outer_train)
    best = None
    for candidate in range(candidates or config.INNER_CANDIDATES):
        train, val = train_test_split(np.arange(len(patients)), test_size=config.INNER_VAL_FRACTION,
            stratify=strata, random_state=(target_seed(target, fold)+candidate+100000) % (2**32-1))
        assignment = {patient: 0 for patient in patients[train]}
        assignment.update({patient: 1 for patient in patients[val]})
        score = distribution_score(outer_train, assignment, [.8, .2])
        if best is None or score < best[0]:
            best = score, assignment, candidate
    return best


def validate_records(records, outer_train_ids=None):
    if records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any():
        raise ValueError("Video duplication or patient leakage")
    if records.groupby("clinical_event_id").hospital_id.nunique().gt(1).any():
        raise ValueError("Clinical event spans patient identities")
    if outer_train_ids is not None and set(records.hospital_id) != set(outer_train_ids):
        raise ValueError("Inner records must contain ONLY outer training patients")


def prepare(source=None, root=None, candidates=None):
    source, root = source or read_records, root or config.SPLITS
    records = {target: source(target).drop(columns=["robust_scaled_raw_value"], errors="ignore").copy() for target in config.TARGETS}
    sources = {target: hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest() for target, frame in records.items()}
    policy = {"version": 1, "seed": config.SEED, "folds": 5,
        "outer_candidates": candidates or config.OUTER_CANDIDATES,
        "inner_candidates": candidates or config.INNER_CANDIDATES,
        "source_records": sources, "inner_val_fraction_of_outer_train": config.INNER_VAL_FRACTION,
        "outer_train_fraction": .8, "outer_test_fraction": .2,
        "test_selection": False, "refit_on_full_outer_train": True}
    manifest = root / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved["policy"] != policy:
            raise RuntimeError("Five-fold sources or split policy changed")
        for name, digest in saved["files"].items():
            if not (root/name).exists() or sha256(root/name) != digest:
                raise RuntimeError(f"Missing/changed fold file: {name}")
        return saved
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Partial split preparation; use a clean split directory")
    root.mkdir(parents=True, exist_ok=True)
    assignments, summaries, choices, files = [], [], {}, {}
    for target, frame in records.items():
        validate_records(frame)
        score, outer, selected = choose_outer(frame, target, candidates)
        choices[target] = {"outer_objective": score, "outer_candidate": selected, "inner": {}}
        assignments.extend({"target": target, "hospital_id": patient, "test_fold": int(fold)} for patient, fold in outer.items())
        for fold in range(5):
            test_mask = frame.hospital_id.map(outer).eq(fold)
            train = frame.loc[~test_mask].copy()
            inner_score, inner, candidate = choose_inner(train, target, fold, candidates)
            choices[target]["inner"][str(fold)] = {"objective": inner_score, "candidate": candidate}
            selection = train.assign(split=np.where(train.hospital_id.map(inner).eq(1), "val", "train"))
            refit = frame.assign(split=np.where(test_mask, "test", "train"))
            validate_records(selection, train.hospital_id.unique())
            validate_records(refit)
            for phase, prepared in (("selection", selection), ("refit", refit)):
                name = f"{target}_fold{fold}_{phase}.csv"
                prepared.to_csv(root/name, index=False)
                files[name] = sha256(root/name)
                for split, group in prepared.groupby("split"):
                    summaries.append({"target": target, "fold": fold, "phase": phase, "split": split,
                        "patients": group.hospital_id.nunique(), "videos": len(group),
                        "clinical_events": group.clinical_event_id.nunique(), "raw_mean": group.raw_value.mean(),
                        "raw_sd": group.raw_value.std(), "raw_q10": group.raw_value.quantile(.1),
                        "raw_median": group.raw_value.median(), "raw_q90": group.raw_value.quantile(.9)})
    for name, frame in (("patient_folds.csv", pd.DataFrame(assignments)), ("distribution_summary.csv", pd.DataFrame(summaries))):
        frame.to_csv(root/name, index=False)
        files[name] = sha256(root/name)
    saved = {"policy": policy, "choices": choices, "files": files,
        "split_objective": "worst normalized Wasserstein + KS + quantile/size discrepancy; no predictions or model metrics",
        "strata": "patient median raw value, rank-quantile strata; all videos of a patient stay together"}
    manifest.write_text(json.dumps(saved, indent=2)+"\n")
    return saved


def read_fold(target, fold, phase):
    return pd.read_csv(config.SPLITS/f"{target}_fold{fold}_{phase}.csv",
        dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
