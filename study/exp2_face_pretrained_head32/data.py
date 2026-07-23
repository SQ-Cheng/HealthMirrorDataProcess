"""Conflict-safe task preparation and 224x224 ImageNet frame loading."""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from .config import (
    BRIGHTNESS_DELTA,
    CONTRAST_DELTA,
    CROP_SCALE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MIN_PATIENTS_PER_CLASS,
    MIN_VIDEOS_PER_CLASS,
    SEED,
    TARGETS,
    VIEW_NAMES,
)


def validate_source_data(source_dir):
    required = ("features.npz", "manifest.csv", "data_quality_report.json")
    missing = [name for name in required if not os.path.exists(os.path.join(source_dir, name))]
    if missing:
        raise FileNotFoundError(f"Missing corrected Exp2 source files: {missing}")
    quality_path = os.path.join(source_dir, "data_quality_report.json")
    with open(quality_path, encoding="utf-8") as handle:
        quality = json.load(handle)
    timezone = quality.get("lab_report_time", {}).get("source_timezone")
    if timezone != "Asia/Shanghai":
        raise RuntimeError(
            f"Source dataset does not declare the corrected Asia/Shanghai time basis: {timezone}"
        )
    hemoglobin_policy = quality.get("hemoglobin_conflicting_video_policy", {})
    required_targets = {"hemoglobin_low", "hemoglobin_moderate_low"}
    if (
        hemoglobin_policy.get("scope") != "per target"
        or not required_targets.issubset(hemoglobin_policy.get("targets", []))
    ):
        raise RuntimeError("Source dataset does not declare the corrected Hb conflict policy")
    return quality


def _conflicting_video_audit(events, target):
    rows = []
    for video_id, group in events.groupby("video_id", sort=True):
        labels = pd.to_numeric(group[target], errors="coerce").dropna().astype(int)
        positive_count = int(labels.eq(1).sum())
        negative_count = int(labels.eq(0).sum())
        if positive_count and negative_count:
            rows.append({
                "target": target,
                "video_id": video_id,
                "hospital_id": str(group["hospital_id"].iloc[0]),
                "positive_event_count": positive_count,
                "negative_event_count": negative_count,
                "event_count": int(len(group)),
                "action": "excluded_for_this_target",
            })
    return rows


def build_task_records(manifest, target):
    labels = pd.to_numeric(manifest[target], errors="coerce")
    invalid_labels = sorted(set(labels.dropna().unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Non-binary values for {target}: {invalid_labels}")
    valid_rows = manifest.loc[labels.notna()].copy()
    valid_rows[target] = labels.loc[valid_rows.index].astype(int)
    if valid_rows.empty:
        return pd.DataFrame(), [], {
            "target": target,
            "status": "skipped",
            "reason": "no non-missing labels",
        }

    events = valid_rows.drop_duplicates("base_event_id").copy()
    audit_rows = _conflicting_video_audit(events, target)
    conflicting_videos = {row["video_id"] for row in audit_rows}
    clean_rows = valid_rows.loc[~valid_rows["video_id"].isin(conflicting_videos)].copy()

    remaining_nunique = clean_rows.groupby("video_id")[target].nunique()
    if not remaining_nunique.le(1).all():
        raise AssertionError(f"Conflicting labels remain for {target}")

    before_deduplication = len(clean_rows)
    records = clean_rows.sort_values(
        ["hospital_id", "video_id", "frame_index", "base_event_id"]
    ).drop_duplicates(["video_id", "frame_index"], keep="first")
    records = records[
        [
            "hospital_id",
            "video_id",
            "video_index",
            "frame_index",
            target,
        ]
    ].reset_index(drop=True)
    records = records.rename(columns={target: "label"})
    frame_counts = records.groupby("video_id")["frame_index"].nunique()
    if not frame_counts.eq(20).all():
        invalid = frame_counts.loc[~frame_counts.eq(20)].to_dict()
        raise ValueError(f"Expected exactly 20 unique frames per video for {target}: {invalid}")
    video_labels = records.drop_duplicates("video_id")
    positive_videos = int(video_labels["label"].eq(1).sum())
    negative_videos = int(video_labels["label"].eq(0).sum())
    summary = {
        "target": target,
        "status": "pending",
        "reason": "",
        "source_events": int(len(events)),
        "source_videos": int(events["video_id"].nunique()),
        "conflicting_videos": int(len(conflicting_videos)),
        "clean_videos": int(len(video_labels)),
        "clean_patients": int(video_labels["hospital_id"].nunique()),
        "positive_videos": positive_videos,
        "negative_videos": negative_videos,
        "unique_frame_rows": int(len(records)),
        "redundant_frame_rows_removed": int(before_deduplication - len(records)),
    }
    if min(positive_videos, negative_videos) < MIN_VIDEOS_PER_CLASS:
        summary.update({
            "status": "skipped",
            "reason": (
                f"fewer than {MIN_VIDEOS_PER_CLASS} clean videos in one class: "
                f"positive={positive_videos}, negative={negative_videos}"
            ),
        })
    return records, audit_rows, summary


def add_patient_split(records, target, seed=SEED):
    video_labels = records.drop_duplicates("video_id")
    patient_labels = video_labels.groupby("hospital_id")["label"].max().reset_index()
    class_counts = patient_labels["label"].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < MIN_PATIENTS_PER_CLASS:
        return None, (
            f"fewer than {MIN_PATIENTS_PER_CLASS} stratifiable patients in one class: "
            f"{class_counts.to_dict()}"
        )
    try:
        first = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
        train_index, temporary_index = next(
            first.split(patient_labels["hospital_id"], patient_labels["label"])
        )
        temporary = patient_labels.iloc[temporary_index].reset_index(drop=True)
        second = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
        validation_index, test_index = next(
            second.split(temporary["hospital_id"], temporary["label"])
        )
    except ValueError as exc:
        return None, f"patient stratification failed: {exc}"

    split_by_patient = {}
    for split_name, frame in (
        ("train", patient_labels.iloc[train_index]),
        ("val", temporary.iloc[validation_index]),
        ("test", temporary.iloc[test_index]),
    ):
        split_by_patient.update({str(value): split_name for value in frame["hospital_id"]})
    result = records.copy()
    result["hospital_id"] = result["hospital_id"].astype(str)
    result["split"] = result["hospital_id"].map(split_by_patient)
    if result["split"].isna().any():
        raise AssertionError(f"Unassigned patients for {target}")
    patient_sets = {
        name: set(result.loc[result["split"].eq(name), "hospital_id"])
        for name in ("train", "val", "test")
    }
    if (
        patient_sets["train"] & patient_sets["val"]
        or patient_sets["train"] & patient_sets["test"]
        or patient_sets["val"] & patient_sets["test"]
    ):
        raise AssertionError(f"Patient leakage detected for {target}")
    return result.reset_index(drop=True), ""


def prepare_tasks(manifest, output_dir, targets=TARGETS, seed=SEED):
    os.makedirs(os.path.join(output_dir, "task_records"), exist_ok=True)
    summaries, conflicts, task_records = [], [], {}
    for target in targets:
        records, audit_rows, summary = build_task_records(manifest, target)
        conflicts.extend(audit_rows)
        if summary["status"] != "skipped":
            records, reason = add_patient_split(records, target, seed)
            if records is None:
                summary.update({"status": "skipped", "reason": reason})
            else:
                summary["status"] = "ready"
                summary["train_patients"] = int(
                    records.loc[records["split"].eq("train"), "hospital_id"].nunique()
                )
                summary["val_patients"] = int(
                    records.loc[records["split"].eq("val"), "hospital_id"].nunique()
                )
                summary["test_patients"] = int(
                    records.loc[records["split"].eq("test"), "hospital_id"].nunique()
                )
                records.to_csv(
                    os.path.join(output_dir, "task_records", f"{target}.csv"), index=False
                )
                task_records[target] = records
        summaries.append(summary)
    summary_frame = pd.DataFrame(summaries)
    conflict_frame = pd.DataFrame(conflicts)
    summary_frame.to_csv(os.path.join(output_dir, "task_summary.csv"), index=False)
    conflict_frame.to_csv(os.path.join(output_dir, "conflicting_videos.csv"), index=False)
    return task_records, summary_frame, conflict_frame


class PretrainedFrameDataset(Dataset):
    def __init__(self, face, records, views=("original",), interpolation="bilinear"):
        self.face = face
        self.records = records.reset_index(drop=True)
        self.video_indices = self.records["video_index"].to_numpy(np.int64)
        self.frame_indices = self.records["frame_index"].to_numpy(np.int64)
        self.labels = self.records["label"].to_numpy(np.float32)
        self.views = tuple(views)
        self.interpolation = interpolation
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.records) * len(self.views)

    def __getitem__(self, index):
        record_index = index // len(self.views)
        view = self.views[index % len(self.views)]
        image = torch.from_numpy(
            self.face[self.video_indices[record_index], self.frame_indices[record_index]]
        ).float().div_(255.0)
        if view == "hflip":
            image = torch.flip(image, dims=(-1,))
        elif view == "center_crop":
            height, width = image.shape[-2:]
            crop_h = max(8, int(round(height * CROP_SCALE)))
            crop_w = max(8, int(round(width * CROP_SCALE)))
            top, left = (height - crop_h) // 2, (width - crop_w) // 2
            image = image[:, top:top + crop_h, left:left + crop_w]
        elif view == "brightness":
            image = (image * (1.0 + BRIGHTNESS_DELTA)).clamp_(0.0, 1.0)
        elif view == "contrast":
            spatial_mean = image.mean(dim=(-2, -1), keepdim=True)
            image = (
                (image - spatial_mean) * (1.0 + CONTRAST_DELTA) + spatial_mean
            ).clamp_(0.0, 1.0)
        elif view != "original":
            raise ValueError(f"Unknown augmentation view: {view}")

        image = functional.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode=self.interpolation,
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        image = (image - self.mean) / self.std
        return (
            image,
            torch.tensor(self.labels[record_index], dtype=torch.float32),
            torch.tensor(record_index, dtype=torch.long),
        )
