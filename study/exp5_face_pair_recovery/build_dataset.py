"""Build patient-disjoint CABG face pairs and trajectory-derived recovery labels."""

from collections import Counter
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from study.exp2_lab_multimodal.build_dataset import (
    _normalize_hospital_id,
    _read_merged_patient_info,
)
from study.exp4.build_dataset import _session_timestamp, _unix_to_local_naive, _video_bounds

from .config import (
    ANALYTES,
    DATA_ROOT,
    LAB_CSV,
    LAB_MATCH_MAX_HOURS,
    MAX_TIME_SOURCE_DELTA_SECONDS,
    SEED,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_SCORE_BINS,
    TIMEZONE,
    TRAJECTORY_BINS,
    TRAJECTORY_GRID_SIZE,
    TRAJECTORY_MIN_SCALE,
    TRAJECTORY_TIME_SCALE,
)


META_COLUMNS = (
    "首页病案号", "首页入院时间", "首页出院时间", "手术开始日期",
    "手术结束日期", "首页手术操作名称",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value):
    return [token.strip() for token in str(value).split("^")]


def _numeric(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(
        cleaned.str.extract(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")[0],
        errors="coerce",
    )


def load_cabg_episodes():
    raw = pd.read_csv(
        LAB_CSV, dtype=str, keep_default_na=False, usecols=list(META_COLUMNS)
    ).drop_duplicates()
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["admission_time"] = pd.to_datetime(raw["首页入院时间"], errors="coerce")
    raw["discharge_time"] = pd.to_datetime(raw["首页出院时间"], errors="coerce")
    rows, audit = [], []
    for episode in raw.itertuples(index=False):
        starts, ends, names = (
            _tokens(getattr(episode, column))
            for column in ("手术开始日期", "手术结束日期", "首页手术操作名称")
        )
        valid_cabg = []
        for position in range(max(len(starts), len(ends), len(names))):
            name = names[position] if position < len(names) else ""
            start = pd.to_datetime(starts[position] if position < len(starts) else "", errors="coerce")
            end = pd.to_datetime(ends[position] if position < len(ends) else "", errors="coerce")
            valid = bool(
                episode.hospital_id and pd.notna(start) and pd.notna(end)
                and pd.notna(episode.admission_time) and pd.notna(episode.discharge_time)
                and end > start and start >= episode.admission_time
                and end <= episode.discharge_time
            )
            is_cabg = "冠状动脉旁路移植" in name
            audit.append({
                "hospital_id": episode.hospital_id,
                "admission_time": episode.admission_time,
                "discharge_time": episode.discharge_time,
                "position": position, "surgery_name": name,
                "surgery_start": start, "surgery_end": end,
                "valid_event": valid, "is_cabg": is_cabg,
            })
            if valid and is_cabg:
                valid_cabg.append((position, start, end, name))
        if valid_cabg:
            position, start, end, name = min(valid_cabg, key=lambda item: item[0])
            rows.append({
                "hospital_id": episode.hospital_id,
                "admission_time": episode.admission_time,
                "discharge_time": episode.discharge_time,
                "surgery_position": position, "surgery_name": name,
                "surgery_start": start, "surgery_end": end,
            })
    episodes = pd.DataFrame(rows).drop_duplicates(
        ["hospital_id", "admission_time", "discharge_time"]
    )
    return episodes.reset_index(drop=True), pd.DataFrame(audit)


def build_video_inventory(episodes):
    mappings = _read_merged_patient_info()
    by_patient = {key: value for key, value in episodes.groupby("hospital_id")}
    rows, audit, counts = [], [], Counter()
    paths = sorted(glob.glob(str(DATA_ROOT / "mirror*_data/patient_*/video.avi")))
    for path_text in paths:
        match = re.search(r"/(mirror\d+)_data/patient_(\d+)/video\.avi$", path_text)
        if match is None:
            continue
        mirror, local_id = match.group(1), int(match.group(2))
        video_id = f"{mirror}_patient_{local_id:06d}"
        mapping = mappings.get((mirror, local_id), {})
        hospital_id = _normalize_hospital_id(mapping.get("Hospital_Patient_ID", ""))
        bounds = _video_bounds(path_text + ".ts")
        session = _session_timestamp(Path(path_text).with_name("patient_info.txt"))
        base = {
            "video_id": video_id, "mirror": mirror, "lab_patient_id": local_id,
            "hospital_id": hospital_id, "video_path": path_text,
        }
        if not hospital_id:
            status = "missing_patient_mapping"
        elif bounds is None or pd.isna(session):
            status = "missing_video_time"
        else:
            start = _unix_to_local_naive(bounds["capture_start_unix"])
            end = _unix_to_local_naive(bounds["capture_end_unix"])
            midpoint = _unix_to_local_naive(
                (bounds["capture_start_unix"] + bounds["capture_end_unix"]) / 2
            )
            base.update(bounds)
            base.update({
                "session_time": session, "capture_start_local": start,
                "capture_end_local": end, "capture_midpoint_local": midpoint,
                "time_source_delta_seconds": abs((session - start).total_seconds()),
            })
            if base["time_source_delta_seconds"] > MAX_TIME_SOURCE_DELTA_SECONDS:
                status = "time_sources_disagree_gt_5min"
            elif hospital_id not in by_patient:
                status = "no_cabg_episode"
            else:
                candidates = by_patient[hospital_id]
                matched = candidates[
                    candidates.admission_time.le(start) & candidates.discharge_time.ge(end)
                ]
                if len(matched) != 1:
                    status = "ambiguous_or_outside_episode"
                else:
                    episode = matched.iloc[0]
                    if end < episode.surgery_start:
                        phase, status = "pre", "retained"
                    elif start >= episode.surgery_end:
                        phase, status = "post", "retained"
                    else:
                        phase, status = "during", "video_overlaps_surgery"
                    if status == "retained":
                        duration = episode.discharge_time - episode.surgery_end
                        progress = (midpoint - episode.surgery_end) / duration
                        rows.append({
                            **base, **episode.to_dict(), "phase": phase,
                            "postoperative_progress": float(progress) if phase == "post" else np.nan,
                            "hours_from_surgery": (midpoint - episode.surgery_end).total_seconds() / 3600,
                        })
        counts[status] += 1
        audit.append({**base, "status": status})
    inventory = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    paired = inventory.groupby("hospital_id").phase.agg(set)
    paired_ids = set(paired[paired.map(lambda phases: {"pre", "post"} <= phases)].index)
    inventory = inventory[inventory.hospital_id.isin(paired_ids)].reset_index(drop=True)
    return inventory, pd.DataFrame(audit), counts


def load_analytes(episodes):
    columns = ["首页病案号", "检验项名称", "检验值(文本)", "单位", "报告时间"]
    raw = pd.read_csv(LAB_CSV, dtype=str, keep_default_na=False, usecols=columns)
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["report_time"] = pd.to_datetime(raw["报告时间"], errors="coerce")
    raw["value"] = _numeric(raw["检验值(文本)"])
    raw["unit_normalized"] = raw["单位"].astype(str).str.replace(r"\s+", "", regex=True).str.lower()
    raw["censored"] = raw["检验值(文本)"].str.match(r"^\s*[<>≤≥＜＞]", na=False)
    frames, audit = [], []
    for analyte, definition in ANALYTES.items():
        selected = raw[
            raw["检验项名称"].eq(definition["item"])
            & raw.unit_normalized.eq(definition["unit"])
        ].copy()
        valid = (
            selected.hospital_id.ne("") & selected.report_time.notna()
            & selected.value.notna() & ~selected.censored
            & selected.value.between(*definition["valid_range"], inclusive="both")
        )
        kept = selected[valid].copy()
        kept["analyte"] = analyte
        kept = kept.groupby(
            ["hospital_id", "report_time", "analyte"], as_index=False
        ).value.median()
        frames.append(kept)
        audit.append({
            "analyte": analyte, "source_item": definition["item"],
            "source_unit": definition["unit"], "source_rows": len(selected),
            "retained_rows": len(kept), "retained_patients": kept.hospital_id.nunique(),
        })
    labs = pd.concat(frames, ignore_index=True)
    labs = labs.merge(
        episodes[["hospital_id", "admission_time", "discharge_time", "surgery_end"]],
        on="hospital_id", how="inner",
    )
    labs = labs[
        labs.report_time.ge(labs.admission_time)
        & labs.report_time.le(labs.discharge_time)
    ].copy()
    duration = (labs.discharge_time - labs.surgery_end).dt.total_seconds()
    labs["postoperative_progress"] = (
        (labs.report_time - labs.surgery_end).dt.total_seconds() / duration
    )
    return labs, pd.DataFrame(audit)


def _nearest(group, start, end):
    if group is None or group.empty:
        return None
    before = (start - group.report_time).dt.total_seconds()
    after = (group.report_time - end).dt.total_seconds()
    interval_delta = np.maximum(np.maximum(before, after), 0.0)
    midpoint = start + (end - start) / 2
    midpoint_delta = (group.report_time - midpoint).abs().dt.total_seconds()
    valid = interval_delta <= LAB_MATCH_MAX_HOURS * 3600
    if not valid.any():
        return None
    candidates = group.loc[valid].copy()
    candidates["interval_delta"] = interval_delta[valid]
    candidates["midpoint_delta"] = midpoint_delta[valid]
    return candidates.sort_values(
        ["interval_delta", "midpoint_delta", "report_time"], kind="stable"
    ).iloc[0]


def match_postoperative_labels(inventory, labs):
    pre = inventory[inventory.phase.eq("pre")].copy()
    post = inventory[inventory.phase.eq("post")].copy()
    pre_lookup = {
        patient: group.sort_values("capture_midpoint_local")
        for patient, group in pre.groupby("hospital_id")
    }
    lab_lookup = {
        (patient, analyte): group.sort_values("report_time")
        for (patient, analyte), group in labs.groupby(["hospital_id", "analyte"])
    }
    rows, exclusions = [], Counter()
    for video in post.itertuples(index=False):
        candidates = pre_lookup[str(video.hospital_id)]
        pre_video = candidates.iloc[
            np.argmin(np.abs((candidates.capture_midpoint_local - video.surgery_start).dt.total_seconds()))
        ]
        row = video._asdict()
        row.update({
            "pre_video_id": pre_video.video_id,
            "pre_video_path": pre_video.video_path,
            "pre_capture_midpoint_local": pre_video.capture_midpoint_local,
            "pre_hours_before_surgery": (
                video.surgery_start - pre_video.capture_midpoint_local
            ).total_seconds() / 3600,
        })
        complete = True
        for analyte in ANALYTES:
            match = _nearest(
                lab_lookup.get((str(video.hospital_id), analyte)),
                video.capture_start_local, video.capture_end_local,
            )
            if match is None:
                exclusions[f"missing_{analyte}"] += 1
                complete = False
                continue
            row[f"{analyte}_value"] = float(match.value)
            row[f"{analyte}_report_time"] = match.report_time
            row[f"{analyte}_signed_delta_hours"] = (
                match.report_time - video.capture_midpoint_local
            ).total_seconds() / 3600
        if complete:
            rows.append(row)
    records = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    return records, exclusions


def _allocation(count):
    train = max(1, round(count * SPLIT_FRACTIONS[0]))
    val = max(1, round(count * SPLIT_FRACTIONS[1]))
    if train + val >= count:
        train = count - val - 1
    return train, val


def add_patient_split(records, seed=SEED):
    patients = records.groupby("hospital_id").postoperative_progress.median().reset_index()
    bins = min(SPLIT_SCORE_BINS, max(3, len(patients) // 12))
    patients["stratum"] = pd.qcut(
        patients.postoperative_progress.rank(method="first"), bins, labels=False
    )
    lookup = {patient: index for index, patient in enumerate(patients.hospital_id)}
    row_patient = records.hospital_id.map(lookup).to_numpy(int)
    variables = ["postoperative_progress"] + [f"{name}_value" for name in ANALYTES]
    scales = {
        name: max(float(records[name].quantile(.75) - records[name].quantile(.25)), 1e-9)
        for name in variables
    }
    rng = np.random.default_rng(seed)
    best = None
    for candidate_index in range(SPLIT_CANDIDATES):
        assignment = np.empty(len(patients), dtype=np.int8)
        for stratum in sorted(patients.stratum.unique()):
            indices = rng.permutation(np.flatnonzero(patients.stratum.to_numpy() == stratum))
            train_n, val_n = _allocation(len(indices))
            assignment[indices[:train_n]] = 0
            assignment[indices[train_n:train_n + val_n]] = 1
            assignment[indices[train_n + val_n:]] = 2
        codes = assignment[row_patient]
        ks_values, wd_values = [], []
        for variable in variables:
            values = records[variable].to_numpy(float)
            for first, second in ((0, 1), (0, 2), (1, 2)):
                ks_values.append(float(ks_2samp(values[codes == first], values[codes == second]).statistic))
                wd_values.append(float(wasserstein_distance(values[codes == first], values[codes == second]) / scales[variable]))
        fractions = np.bincount(codes, minlength=3) / len(codes)
        size_error = float(np.max(np.abs(fractions - np.asarray(SPLIT_FRACTIONS))))
        objective = 2 * max(wd_values) + max(ks_values) + .25 * size_error
        key = (objective, max(wd_values), max(ks_values), candidate_index)
        if best is None or key < best[0]:
            best = (key, assignment.copy(), candidate_index, max(ks_values), max(wd_values), size_error)
    _, assignment, candidate, max_ks, max_wd, size_error = best
    result = records.copy()
    result["split"] = np.asarray(("train", "val", "test"))[assignment[row_patient]]
    if result.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Patient leakage in paired recovery split")
    manifest = {
        "seed": seed, "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate_index": int(candidate), "objective": float(best[0][0]),
        "max_ks": float(max_ks), "max_wasserstein_iqr": float(max_wd),
        "size_fraction_error": size_error, "variables": variables,
    }
    return result, manifest


def fit_trajectories_and_score(records, labs):
    train_patients = set(records.loc[records.split.eq("train"), "hospital_id"])
    source = labs[
        labs.hospital_id.isin(train_patients)
        & labs.postoperative_progress.between(0, 1, inclusive="both")
    ].copy()
    source["trajectory_bin"] = np.minimum(
        (source.postoperative_progress * TRAJECTORY_BINS).astype(int),
        TRAJECTORY_BINS - 1,
    )
    patient_bins = source.groupby(
        ["analyte", "hospital_id", "trajectory_bin"], as_index=False
    ).value.median()
    rows, models = [], {}
    grid = np.linspace(0, 1, TRAJECTORY_GRID_SIZE)
    centers = (np.arange(TRAJECTORY_BINS) + .5) / TRAJECTORY_BINS
    for analyte, definition in ANALYTES.items():
        raw_values = source.loc[source.analyte.eq(analyte), "value"].to_numpy(float)
        if definition.get("log1p"):
            raw_values = np.log1p(raw_values)
        median = float(np.median(raw_values))
        iqr = max(float(np.quantile(raw_values, .75) - np.quantile(raw_values, .25)), 1e-6)
        values = patient_bins[patient_bins.analyte.eq(analyte)].copy()
        if definition.get("log1p"):
            values["model_value"] = np.log1p(values.value)
        else:
            values["model_value"] = values.value
        summary = values.groupby("trajectory_bin").model_value.agg(
            median="median", q25=lambda x: np.quantile(x, .25),
            q75=lambda x: np.quantile(x, .75), patients="size",
        ).reindex(range(TRAJECTORY_BINS))
        for column in ("median", "q25", "q75"):
            summary[column] = summary[column].interpolate(limit_direction="both")
        summary["median"] = summary["median"].rolling(3, center=True, min_periods=1).median()
        curve = np.interp(grid, centers, (summary["median"] - median) / iqr)
        scale = np.interp(grid, centers, (summary.q75 - summary.q25) / iqr)
        scale = np.maximum(scale, TRAJECTORY_MIN_SCALE)
        models[analyte] = {"median": median, "iqr": iqr, "curve": curve, "scale": scale}
        for index, item in summary.iterrows():
            rows.append({
                "analyte": analyte, "trajectory_bin": index,
                "progress_center": centers[index], "median": item["median"],
                "q25": item.q25, "q75": item.q75,
                "patients": int(item.patients) if pd.notna(item.patients) else 0,
                "transform": "log1p" if definition.get("log1p") else "identity",
                "global_train_median": median, "global_train_iqr": iqr,
            })
    result = records.copy()
    component_columns = []
    for analyte, definition in ANALYTES.items():
        model = models[analyte]
        observed = result[f"{analyte}_value"].to_numpy(float)
        if definition.get("log1p"):
            observed = np.log1p(observed)
        z = (observed - model["median"]) / model["iqr"]
        prior = result.postoperative_progress.to_numpy(float)
        distances = (
            ((z[:, None] - model["curve"][None, :]) / model["scale"][None, :]) ** 2
            + ((grid[None, :] - prior[:, None]) / TRAJECTORY_TIME_SCALE) ** 2
        )
        column = f"{analyte}_recovery_component"
        result[column] = grid[np.argmin(distances, axis=1)]
        component_columns.append(column)
    result["recovery_score"] = result[component_columns].mean(axis=1)
    if not result.recovery_score.between(0, 1).all():
        raise AssertionError("Trajectory recovery score outside [0,1]")
    return result, pd.DataFrame(rows)


def prepare_records(output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    episodes, surgery_audit = load_cabg_episodes()
    inventory, video_audit, statuses = build_video_inventory(episodes)
    labs, lab_audit = load_analytes(episodes)
    records, match_exclusions = match_postoperative_labels(inventory, labs)
    records, split_manifest = add_patient_split(records)
    records, trajectories = fit_trajectories_and_score(records, labs)
    pre_ids = set(records.pre_video_id); post_ids = set(records.video_id)
    frame_records = inventory[inventory.video_id.isin(pre_ids | post_ids)][
        ["video_id", "video_path"]
    ].drop_duplicates().reset_index(drop=True)
    records.to_csv(output_dir / "records.csv", index=False)
    frame_records.to_csv(output_dir / "frame_records.csv", index=False)
    trajectories.to_csv(output_dir / "recovery_trajectories.csv", index=False)
    episodes.to_csv(output_dir / "cabg_episodes.csv", index=False)
    surgery_audit.to_csv(output_dir / "surgery_event_audit.csv", index=False)
    video_audit.to_csv(output_dir / "video_eligibility_audit.csv", index=False)
    lab_audit.to_csv(output_dir / "lab_source_audit.csv", index=False)
    split_rows = []
    for split, group in records.groupby("split"):
        split_rows.append({
            "split": split, "videos": len(group), "patients": group.hospital_id.nunique(),
            "score_mean": group.recovery_score.mean(), "score_std": group.recovery_score.std(),
            "score_q10": group.recovery_score.quantile(.1),
            "score_median": group.recovery_score.median(),
            "score_q90": group.recovery_score.quantile(.9),
        })
    pd.DataFrame(split_rows).to_csv(output_dir / "split_distribution.csv", index=False)
    manifest = {
        "schema_version": 1,
        "experiment": "exp5_cabg_pre_post_face_pair_recovery",
        "source_sha256": _sha256(LAB_CSV), "timezone": TIMEZONE,
        "analytes": ANALYTES,
        "label": {
            "method": "equal-weight mean of five trajectory-projected recovery positions",
            "trajectory_fit_scope": "train patients only",
            "projection_time_scale": TRAJECTORY_TIME_SCALE,
            "lab_match_max_hours": LAB_MATCH_MAX_HOURS,
            "requires_all_five_analytes": True,
        },
        "pairing": "nearest available preoperative video from the same CABG hospitalization",
        "counts": {
            "cabg_episodes": len(episodes), "paired_inventory_videos": len(inventory),
            "labelled_postoperative_videos": len(records),
            "labelled_patients": records.hospital_id.nunique(),
            "frame_index_videos": len(frame_records),
            "video_statuses": dict(statuses), "lab_match_exclusions": dict(match_exclusions),
        },
        "split": split_manifest,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[data] records={len(records)} patients={records.hospital_id.nunique()} "
        f"pre_videos={records.pre_video_id.nunique()} post_videos={records.video_id.nunique()} "
        f"split={records.groupby('split').size().to_dict()}", flush=True,
    )
    return records, frame_records, manifest
