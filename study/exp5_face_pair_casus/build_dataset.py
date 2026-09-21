"""Build CABG face pairs labelled with the four-laboratory CASUS subtotal."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from study.exp5_face_pair_recovery.build_dataset import (
    _nearest,
    build_video_inventory,
    load_cabg_episodes,
)
from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id

from .config import (
    CASUS_ANALYTES,
    CASUS_MAX_SCORE,
    LAB_CSV,
    LAB_MATCH_MAX_HOURS,
    SEED,
    SPLIT_CANDIDATES,
    SPLIT_FRACTIONS,
    SPLIT_SCORE_BINS,
    TIMEZONE,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(
        cleaned.str.extract(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")[0],
        errors="coerce",
    )


def _normalized_unit(unit, value_text):
    text = f"{unit} {value_text}".lower().replace(" ", "")
    text = text.replace("µ", "μ").replace("umol", "μmol")
    if "mg/dl" in text:
        return "mg/dl"
    if "μmol/l" in text:
        return "umol/l"
    if "mmol/l" in text:
        return "mmol/l"
    if "10^9/l" in text or "10∧9/l" in text or "g/l" in text:
        return "10^9/l"
    return ""


def _convert_value(analyte, value, unit):
    if pd.isna(value):
        return np.nan
    if analyte in {"creatinine", "bilirubin"}:
        if unit == "mg/dl":
            return float(value)
        if unit == "umol/l":
            divisor = 88.4 if analyte == "creatinine" else 17.104
            return float(value) / divisor
        return np.nan
    if analyte == "lactate":
        return float(value) if unit == "mmol/l" else np.nan
    if analyte == "platelets":
        return float(value) if unit == "10^9/l" else np.nan
    raise KeyError(analyte)


def load_casus_labs(episodes):
    columns = ["首页病案号", "检验套名称", "检验项名称", "检验值(文本)", "单位", "标本名称", "报告时间"]
    raw = pd.read_csv(LAB_CSV, dtype=str, keep_default_na=False, usecols=columns)
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["report_time"] = pd.to_datetime(raw["报告时间"], errors="coerce")
    raw["numeric_value"] = _numeric(raw["检验值(文本)"])
    raw["censored"] = raw["检验值(文本)"].str.match(r"^\s*[<>≤≥＜＞]", na=False)
    raw["source_unit"] = [
        _normalized_unit(unit, value)
        for unit, value in zip(raw["单位"], raw["检验值(文本)"])
    ]

    kept_frames, audit_frames = [], []
    for analyte, definition in CASUS_ANALYTES.items():
        selected = raw[raw["检验项名称"].isin(definition["items"])].copy()
        selected["item_allowed"] = True
        selected["specimen_allowed"] = selected["标本名称"].isin(definition["specimens"])
        selected["canonical_value"] = [
            _convert_value(analyte, value, unit)
            for value, unit in zip(selected.numeric_value, selected.source_unit)
        ]
        selected["unit_supported"] = selected.canonical_value.notna()
        selected["in_valid_range"] = selected.canonical_value.between(
            *definition["valid_range"], inclusive="both"
        )
        selected["valid"] = (
            selected.hospital_id.ne("")
            & selected.report_time.notna()
            & ~selected.censored
            & selected.specimen_allowed
            & selected.unit_supported
            & selected.in_valid_range
        )
        audit = selected.groupby(
            ["检验套名称", "检验项名称", "单位", "source_unit", "标本名称"],
            dropna=False,
        ).agg(
            source_rows=("valid", "size"),
            retained_rows=("valid", "sum"),
            patients=("hospital_id", lambda values: values[values.ne("")].nunique()),
        ).reset_index()
        audit.insert(0, "analyte", analyte)
        audit["canonical_unit"] = definition["canonical_unit"]
        audit_frames.append(audit)

        kept = selected[selected.valid].copy()
        kept["analyte"] = analyte
        kept = kept.groupby(
            ["hospital_id", "report_time", "analyte"], as_index=False
        ).canonical_value.median().rename(columns={"canonical_value": "value"})
        kept_frames.append(kept)

    labs = pd.concat(kept_frames, ignore_index=True)
    labs = labs.merge(
        episodes[["hospital_id", "admission_time", "discharge_time", "surgery_end"]],
        on="hospital_id", how="inner",
    )
    labs = labs[
        labs.report_time.ge(labs.admission_time)
        & labs.report_time.le(labs.discharge_time)
        & labs.report_time.ge(labs.surgery_end)
    ].copy()
    return labs, pd.concat(audit_frames, ignore_index=True)


def _casus_component(analyte, value):
    value = float(value)
    if analyte == "creatinine":
        return int(np.digitize(value, [1.2, 2.3, 4.1, np.nextafter(5.5, np.inf)]))
    if analyte == "bilirubin":
        return int(np.digitize(value, [1.2, 3.6, 7.1, np.nextafter(14.0, np.inf)]))
    if analyte == "lactate":
        return int(np.digitize(value, [2.1, 4.1, 8.1, np.nextafter(12.0, np.inf)]))
    if analyte == "platelets":
        if value > 120:
            return 0
        if value >= 81:
            return 1
        if value >= 51:
            return 2
        if value >= 21:
            return 3
        return 4
    raise KeyError(analyte)


def match_postoperative_labels(inventory, labs, include_preoperative=True):
    pre = inventory[inventory.phase.eq("pre")].copy()
    post = inventory[inventory.phase.eq("post")].copy()
    pre_lookup = {}
    if include_preoperative:
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
        if include_preoperative:
            candidates = pre_lookup[str(video.hospital_id)]
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
        complete = True
        report_times = []
        for analyte in CASUS_ANALYTES:
            match = _nearest(
                lab_lookup.get((str(video.hospital_id), analyte)),
                video.capture_start_local, video.capture_end_local,
            )
            if match is None:
                exclusions[f"missing_{analyte}_within_{LAB_MATCH_MAX_HOURS:g}h"] += 1
                complete = False
                continue
            value = float(match.value)
            row[f"{analyte}_value"] = value
            row[f"{analyte}_casus_points"] = _casus_component(analyte, value)
            row[f"{analyte}_report_time"] = match.report_time
            row[f"{analyte}_signed_delta_hours"] = (
                match.report_time - video.capture_midpoint_local
            ).total_seconds() / 3600
            report_times.append(match.report_time)
        if complete:
            row["casus_score"] = sum(
                row[f"{analyte}_casus_points"] for analyte in CASUS_ANALYTES
            )
            row["lab_time_span_hours"] = (
                max(report_times) - min(report_times)
            ).total_seconds() / 3600
            rows.append(row)
    records = pd.DataFrame(rows).sort_values("video_id").reset_index(drop=True)
    if len(records) and not records.casus_score.between(0, CASUS_MAX_SCORE).all():
        raise AssertionError("CASUS subtotal outside [0, 16]")
    return records, exclusions


def _allocation(count):
    train = max(1, round(count * SPLIT_FRACTIONS[0]))
    val = max(1, round(count * SPLIT_FRACTIONS[1]))
    if train + val >= count:
        train = count - val - 1
    return train, val


def add_patient_split(records, seed=SEED):
    patients = records.groupby("hospital_id").casus_score.median().reset_index()
    bins = min(SPLIT_SCORE_BINS, max(3, len(patients) // 12))
    patients["stratum"] = pd.qcut(
        patients.casus_score.rank(method="first"), bins, labels=False
    )
    lookup = {patient: index for index, patient in enumerate(patients.hospital_id)}
    row_patient = records.hospital_id.map(lookup).to_numpy(int)
    variables = ["casus_score", "postoperative_progress"] + [
        f"{name}_value" for name in CASUS_ANALYTES
    ]
    scales = {
        name: max(float(records[name].quantile(.75) - records[name].quantile(.25)), 1.0e-6)
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
        objective = 2 * max(wd_values) + max(ks_values) + 0.25 * size_error
        key = objective, max(wd_values), max(ks_values), candidate_index
        if best is None or key < best[0]:
            best = key, assignment.copy(), candidate_index, max(ks_values), max(wd_values), size_error
    _, assignment, candidate, max_ks, max_wd, size_error = best
    result = records.copy()
    result["split"] = np.asarray(("train", "val", "test"))[assignment[row_patient]]
    if result.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError("Patient leakage in CASUS split")
    return result, {
        "seed": seed,
        "candidate_count": SPLIT_CANDIDATES,
        "selected_candidate_index": int(candidate),
        "objective": float(best[0][0]),
        "max_ks": float(max_ks),
        "max_wasserstein_iqr": float(max_wd),
        "size_fraction_error": size_error,
        "variables": variables,
    }


def write_label_outputs(records, output_dir):
    split_rows = []
    for split, group in records.groupby("split"):
        split_rows.append({
            "split": split,
            "videos": len(group),
            "patients": group.hospital_id.nunique(),
            "score_mean": group.casus_score.mean(),
            "score_std": group.casus_score.std(),
            "score_q10": group.casus_score.quantile(0.1),
            "score_median": group.casus_score.median(),
            "score_q90": group.casus_score.quantile(0.9),
        })
    pd.DataFrame(split_rows).to_csv(output_dir / "split_distribution.csv", index=False)
    component_rows = []
    for split, group in records.groupby("split"):
        for analyte in CASUS_ANALYTES:
            counts = group[f"{analyte}_casus_points"].value_counts()
            for points in range(5):
                component_rows.append({
                    "split": split,
                    "analyte": analyte,
                    "casus_points": points,
                    "videos": int(counts.get(points, 0)),
                })
    pd.DataFrame(component_rows).to_csv(output_dir / "casus_component_distribution.csv", index=False)


def prepare_records(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, surgery_audit = load_cabg_episodes()
    inventory, video_audit, statuses = build_video_inventory(episodes)
    labs, lab_audit = load_casus_labs(episodes)
    records, match_exclusions = match_postoperative_labels(inventory, labs)
    if records.empty:
        raise RuntimeError("No postoperative videos have all four CASUS laboratory components")
    records, split_manifest = add_patient_split(records)
    pre_ids, post_ids = set(records.pre_video_id), set(records.video_id)
    frame_records = inventory[inventory.video_id.isin(pre_ids | post_ids)][
        ["video_id", "video_path"]
    ].drop_duplicates().reset_index(drop=True)

    records.to_csv(output_dir / "records.csv", index=False)
    frame_records.to_csv(output_dir / "frame_records.csv", index=False)
    episodes.to_csv(output_dir / "cabg_episodes.csv", index=False)
    surgery_audit.to_csv(output_dir / "surgery_event_audit.csv", index=False)
    video_audit.to_csv(output_dir / "video_eligibility_audit.csv", index=False)
    lab_audit.to_csv(output_dir / "lab_source_audit.csv", index=False)
    write_label_outputs(records, output_dir)
    manifest = {
        "schema_version": 1,
        "experiment": "exp5_cabg_pre_post_face_pair_partial_casus",
        "source_sha256": _sha256(LAB_CSV),
        "timezone": TIMEZONE,
        "label": {
            "name": "four-laboratory partial-CASUS subtotal",
            "range_points": [0, 16],
            "components": list(CASUS_ANALYTES),
            "lab_match_max_hours": LAB_MATCH_MAX_HOURS,
            "requires_all_four_components": True,
            "not_full_casus": True,
            "reference": "https://academic.oup.com/ejcts/article/38/1/104/469021",
        },
        "analyte_sources": CASUS_ANALYTES,
        "pairing": "nearest available preoperative video from the same CABG hospitalization",
        "counts": {
            "cabg_episodes": len(episodes),
            "paired_inventory_videos": len(inventory),
            "labelled_postoperative_videos": len(records),
            "labelled_patients": records.hospital_id.nunique(),
            "frame_index_videos": len(frame_records),
            "video_statuses": dict(statuses),
            "lab_match_exclusions": dict(match_exclusions),
        },
        "split": split_manifest,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[data] records={len(records)} patients={records.hospital_id.nunique()} "
        f"pre_videos={records.pre_video_id.nunique()} post_videos={records.video_id.nunique()} "
        f"score={records.casus_score.min():.0f}-{records.casus_score.max():.0f} "
        f"split={records.groupby('split').size().to_dict()}",
        flush=True,
    )
    return records, frame_records, manifest


def prepare_post_only_records(output_dir):
    """Build CASUS labels from every eligible postoperative video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes, surgery_audit = load_cabg_episodes()
    inventory, video_audit, statuses = build_video_inventory(
        episodes, require_paired=False,
    )
    labs, lab_audit = load_casus_labs(episodes)
    records, match_exclusions = match_postoperative_labels(
        inventory, labs, include_preoperative=False,
    )
    if records.empty:
        raise RuntimeError("No postoperative videos have all four CASUS components")
    records, split_manifest = add_patient_split(records)
    frame_records = records[["video_id", "video_path"]].drop_duplicates().reset_index(drop=True)

    records.to_csv(output_dir / "records.csv", index=False)
    frame_records.to_csv(output_dir / "frame_records.csv", index=False)
    episodes.to_csv(output_dir / "cabg_episodes.csv", index=False)
    surgery_audit.to_csv(output_dir / "surgery_event_audit.csv", index=False)
    video_audit.to_csv(output_dir / "video_eligibility_audit.csv", index=False)
    lab_audit.to_csv(output_dir / "lab_source_audit.csv", index=False)
    write_label_outputs(records, output_dir)
    manifest = {
        "schema_version": 1,
        "experiment": "exp5_cabg_post_face_only_partial_casus",
        "source_sha256": _sha256(LAB_CSV),
        "timezone": TIMEZONE,
        "label": {
            "name": "four-laboratory partial-CASUS subtotal",
            "range_points": [0, 16],
            "components": list(CASUS_ANALYTES),
            "lab_match_max_hours": LAB_MATCH_MAX_HOURS,
            "requires_all_four_components": True,
            "not_full_casus": True,
            "reference": "https://academic.oup.com/ejcts/article/38/1/104/469021",
        },
        "analyte_sources": CASUS_ANALYTES,
        "video_selection": (
            "all retained postoperative CABG videos; no preoperative video required"
        ),
        "counts": {
            "cabg_episodes": len(episodes),
            "retained_inventory_videos": len(inventory),
            "retained_postoperative_inventory_videos": int(inventory.phase.eq("post").sum()),
            "labelled_postoperative_videos_before_frame_validation": len(records),
            "labelled_patients_before_frame_validation": records.hospital_id.nunique(),
            "frame_index_requested_videos": len(frame_records),
            "video_statuses": dict(statuses),
            "lab_match_exclusions": dict(match_exclusions),
        },
        "split": split_manifest,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[data] protocol=post_face_only records={len(records)} "
        f"patients={records.hospital_id.nunique()} "
        f"post_inventory={int(inventory.phase.eq('post').sum())} "
        f"score={records.casus_score.min():.0f}-{records.casus_score.max():.0f} "
        f"split={records.groupby('split').size().to_dict()}",
        flush=True,
    )
    return records, frame_records, manifest
