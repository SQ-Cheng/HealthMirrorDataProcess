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

from study.common.time_alignment import (
    TimeAlignmentError,
    read_video_session,
    unix_to_local_naive,
)
from study.exp2_lab_multimodal.build_dataset import (
    _normalize_hospital_id,
    _read_merged_patient_info,
)
from study.exp2_lab_longitudinal_statistics.run_analysis import _load_and_clean

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
    TARGET_COLUMN,
    TRAJECTORY_BINS,
    TRAJECTORY_GRID_SIZE,
    TRAJECTORY_MIN_SCALE,
)


META_COLUMNS = (
    "首页病案号", "首页入院时间", "首页出院时间", "手术开始日期",
    "手术结束日期", "首页手术操作名称",
)
PROTOCOLS = ("paired", "pre_only", "post_only")


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


def build_video_inventory(episodes, require_paired=True):
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
        base = {
            "video_id": video_id, "mirror": mirror, "lab_patient_id": local_id,
            "hospital_id": hospital_id, "video_path": path_text,
        }
        if not hospital_id:
            status = "missing_patient_mapping"
        else:
            try:
                timing = read_video_session(
                    path_text,
                    expected_local_id=local_id,
                    expected_hospital_id=hospital_id,
                    timezone=TIMEZONE,
                )
            except TimeAlignmentError as exc:
                status = exc.code
                base["validation_error"] = str(exc)
                timing = None
            if timing is not None:
                start = unix_to_local_naive(timing["capture_start_unix"], TIMEZONE)
                end = unix_to_local_naive(timing["capture_end_unix"], TIMEZONE)
                midpoint = unix_to_local_naive(timing["capture_midpoint_unix"], TIMEZONE)
                base.update(timing)
                base.update({
                    "session_time": timing["session_time_local"],
                    "capture_start_local": start,
                    "capture_end_local": end,
                    "capture_midpoint_local": midpoint,
                    "time_source_delta_seconds": timing[
                        "frame_timestamp_source_abs_delta_seconds"
                    ],
                })
            if timing is None:
                pass
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
    if require_paired:
        paired = inventory.groupby("hospital_id").phase.agg(set)
        paired_ids = set(
            paired[paired.map(lambda phases: {"pre", "post"} <= phases)].index
        )
        inventory = inventory[inventory.hospital_id.isin(paired_ids)].reset_index(drop=True)
    return inventory, pd.DataFrame(audit), counts


def load_analytes(episodes):
    _, measurements, _, _, _, harmonization_audit, _ = _load_and_clean(LAB_CSV)
    measurements = measurements.merge(
        episodes[[
            "hospital_id", "admission_time", "discharge_time",
            "surgery_start", "surgery_end",
        ]],
        on=["hospital_id", "admission_time", "discharge_time"], how="inner",
        validate="many_to_one",
    )
    frames, audit = [], []
    for analyte, definition in ANALYTES.items():
        selected = measurements[
            measurements.item_name.eq(definition["item"])
            & measurements.unit.eq(definition["unit"])
        ].copy()
        selected = selected.rename(columns={"numeric_value": "value"})
        preoperative = selected.report_time.lt(selected.surgery_start)
        intraoperative = (
            selected.report_time.ge(selected.surgery_start)
            & selected.report_time.le(selected.surgery_end)
        )
        postoperative = selected.report_time.gt(selected.surgery_end)
        valid = (
            postoperative
            & selected.value.between(*definition["valid_range"], inclusive="both")
        )
        kept = selected.loc[valid, [
            "hospital_id", "admission_time", "discharge_time", "surgery_start",
            "surgery_end", "report_time", "value", "source_item_names",
            "source_units", "harmonization_rules",
        ]].copy()
        kept["analyte"] = analyte
        kept = kept.groupby(
            [
                "hospital_id", "admission_time", "discharge_time", "surgery_start",
                "surgery_end", "report_time", "analyte",
            ], as_index=False
        ).agg(
            value=("value", "median"),
            source_item_names=("source_item_names", lambda x: "^".join(sorted(set(x)))),
            source_units=("source_units", lambda x: "^".join(sorted(set(x)))),
            harmonization_rules=("harmonization_rules", lambda x: "^".join(sorted(set(x)))),
        )
        frames.append(kept)
        audit.append({
            "analyte": analyte, "source_item": definition["item"],
            "source_unit": definition["unit"], "harmonized_source_rows": len(selected),
            "preoperative_rows_excluded": int(preoperative.sum()),
            "intraoperative_rows_excluded": int(intraoperative.sum()),
            "postoperative_out_of_range_rows_excluded": int(
                (postoperative & ~selected.value.between(
                    *definition["valid_range"], inclusive="both"
                )).sum()
            ),
            "retained_rows": len(kept), "retained_patients": kept.hospital_id.nunique(),
        })
    labs = pd.concat(frames, ignore_index=True)
    duration = (labs.discharge_time - labs.surgery_end).dt.total_seconds()
    labs["postoperative_progress"] = (
        (labs.report_time - labs.surgery_end).dt.total_seconds() / duration
    )
    return labs, pd.DataFrame(audit), harmonization_audit


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


def match_postoperative_labels(inventory, labs, protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
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
        row = video._asdict()
        candidates = pre_lookup.get(str(video.hospital_id))
        if protocol != "post_only" and (candidates is None or candidates.empty):
            exclusions["missing_preoperative_video"] += 1
            continue
        if candidates is not None and not candidates.empty:
            pre_video = candidates.iloc[
                np.argmin(np.abs((candidates.capture_midpoint_local - video.surgery_start).dt.total_seconds()))
            ]
            row.update({
                "pre_video_id": pre_video.video_id,
                "pre_video_path": pre_video.video_path,
                "pre_capture_midpoint_local": pre_video.capture_midpoint_local,
                "pre_hours_before_surgery": (
                    video.surgery_start - pre_video.capture_midpoint_local
                ).total_seconds() / 3600,
            })
        else:
            row.update({
                "pre_video_id": "", "pre_video_path": "",
                "pre_capture_midpoint_local": pd.NaT,
                "pre_hours_before_surgery": np.nan,
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
        if not len(raw_values):
            raise ValueError(f"No training-patient trajectory values for {analyte}")
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
        progress = np.clip(result.postoperative_progress.to_numpy(float), 0.0, 1.0)
        expected = np.interp(progress, grid, model["curve"])
        local_scale = np.interp(progress, grid, model["scale"])
        column = f"{analyte}_deviation_component"
        result[column] = np.abs(z - expected) / local_scale
        component_columns.append(column)
    result[TARGET_COLUMN] = result[component_columns].mean(axis=1)
    if not np.isfinite(result[TARGET_COLUMN]).all() or result[TARGET_COLUMN].lt(0).any():
        raise AssertionError("Invalid postoperative trajectory-deviation score")
    return result, pd.DataFrame(rows)


def protocol_output_dir(output_dir, protocol):
    output_dir = Path(output_dir)
    return output_dir if protocol == "paired" else output_dir / "ablations" / protocol


def prepare_candidates(output_dir):
    """Load shared sources once and construct maximum candidate cohorts."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        output_dir / "figures" / "recovery_score_definition.png",
        output_dir / "figures" / "gradcam_occlusion_10_faces.png",
    ):
        stale.unlink(missing_ok=True)
    episodes, surgery_audit = load_cabg_episodes()
    inventory, video_audit, statuses = build_video_inventory(episodes, require_paired=False)
    labs, lab_audit, harmonization_audit = load_analytes(episodes)
    candidates, exclusions = {}, {}
    requested_ids = set()
    for protocol in PROTOCOLS:
        records, match_exclusions = match_postoperative_labels(inventory, labs, protocol)
        candidates[protocol] = records
        exclusions[protocol] = dict(match_exclusions)
        requested_ids.update(records.video_id.astype(str))
        if protocol != "post_only":
            requested_ids.update(records.pre_video_id.astype(str))
    frame_records = inventory[inventory.video_id.astype(str).isin(requested_ids)][
        ["video_id", "video_path"]
    ].drop_duplicates().sort_values("video_id").reset_index(drop=True)
    episodes.to_csv(output_dir / "cabg_episodes.csv", index=False)
    surgery_audit.to_csv(output_dir / "surgery_event_audit.csv", index=False)
    video_audit.to_csv(output_dir / "video_eligibility_audit.csv", index=False)
    lab_audit.to_csv(output_dir / "lab_source_audit.csv", index=False)
    harmonization_audit.to_csv(output_dir / "lab_harmonization_audit.csv", index=False)
    frame_records.to_csv(output_dir / "frame_records.csv", index=False)
    shared = {
        "source_sha256": _sha256(LAB_CSV), "timezone": TIMEZONE,
        "video_statuses": dict(statuses), "lab_match_exclusions": exclusions,
        "cabg_episodes": len(episodes), "inventory_videos": len(inventory),
    }
    return candidates, frame_records, labs, shared


def finalize_protocol_records(protocol, records, labs, usable_video_ids, output_dir, shared):
    """Apply only the face availability required by one protocol, then split and score."""
    run_dir = protocol_output_dir(output_dir, protocol)
    run_dir.mkdir(parents=True, exist_ok=True)
    usable = set(map(str, usable_video_ids))
    post_ok = records.video_id.astype(str).isin(usable)
    pre_ok = records.pre_video_id.astype(str).isin(usable)
    if protocol == "paired":
        keep = post_ok & pre_ok
    elif protocol == "pre_only":
        keep = pre_ok
    else:
        keep = post_ok
    invalid = records.loc[~keep].copy()
    invalid["post_frame_required"] = protocol in {"paired", "post_only"}
    invalid["post_frame_usable"] = post_ok.loc[~keep].to_numpy()
    invalid["pre_frame_required"] = protocol in {"paired", "pre_only"}
    invalid["pre_frame_usable"] = pre_ok.loc[~keep].to_numpy()
    invalid.to_csv(run_dir / "frame_exclusions.csv", index=False)
    records = records.loc[keep].copy().reset_index(drop=True)
    if records.hospital_id.nunique() < 15:
        raise RuntimeError(
            f"Too few patients after {protocol} face validation: "
            f"{records.hospital_id.nunique()}"
        )
    records, split_manifest = add_patient_split(records)
    records, trajectories = fit_trajectories_and_score(records, labs)
    records.to_csv(run_dir / "records.csv", index=False)
    trajectories.to_csv(run_dir / "recovery_trajectories.csv", index=False)
    split_rows = []
    for split, group in records.groupby("split"):
        split_rows.append({
            "split": split, "videos": len(group), "patients": group.hospital_id.nunique(),
            "score_mean": group[TARGET_COLUMN].mean(), "score_std": group[TARGET_COLUMN].std(),
            "score_q10": group[TARGET_COLUMN].quantile(.1),
            "score_median": group[TARGET_COLUMN].median(),
            "score_q90": group[TARGET_COLUMN].quantile(.9),
        })
    pd.DataFrame(split_rows).to_csv(run_dir / "split_distribution.csv", index=False)
    manifest = {
        "schema_version": 2,
        "experiment": "exp5_cabg_face_recovery", "protocol": protocol,
        "source_sha256": shared["source_sha256"], "timezone": TIMEZONE,
        "analytes": ANALYTES,
        "label": {
            "method": (
                "equal-weight mean of nine absolute robust-standardized residuals "
                "from the train-patient average postoperative trajectory at video time"
            ),
            "trajectory_fit_scope": "train patients only",
            "lab_match_max_hours": LAB_MATCH_MAX_HOURS,
            "requires_all_nine_analytes": True,
            "temporal_policy": (
                "all analytes use reports strictly after surgery_end only; preoperative "
                "and intraoperative reports are excluded before matching and fitting"
            ),
        },
        "input_requirement": {
            "paired": "usable preoperative and postoperative face videos",
            "pre_only": "usable preoperative face video only",
            "post_only": "usable postoperative face video only",
        }[protocol],
        "pairing": (
            "nearest available preoperative video from the same CABG hospitalization"
            if protocol != "post_only" else "not applicable"
        ),
        "counts": {
            "cabg_episodes": shared["cabg_episodes"],
            "all_phase_inventory_videos": shared["inventory_videos"],
            "candidate_postoperative_videos": len(records) + len(invalid),
            "frame_excluded_records": len(invalid),
            "labelled_postoperative_videos": len(records),
            "labelled_patients": records.hospital_id.nunique(),
            "video_statuses": shared["video_statuses"],
            "lab_match_exclusions": shared["lab_match_exclusions"][protocol],
        },
        "split": split_manifest,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[data] protocol={protocol} records={len(records)} "
        f"patients={records.hospital_id.nunique()} "
        f"pre_videos={records.pre_video_id.nunique()} post_videos={records.video_id.nunique()} "
        f"split={records.groupby('split').size().to_dict()}", flush=True,
    )
    return records, manifest
