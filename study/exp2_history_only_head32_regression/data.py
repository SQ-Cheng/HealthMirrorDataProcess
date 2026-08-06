"""One history sequence per video using the exact reference task records."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from study.exp2_face_history_head32_regression.history_data import HistoryFeatureStore


class HistoryOnlyDataset(Dataset):
    def __init__(self, records, history_store):
        self.records = records.reset_index(drop=True).copy()
        if self.records["video_id"].astype(str).duplicated().any():
            raise ValueError("History-only data requires one row per video")
        history_lookup = history_store.lookup()
        sequence_length = max(1, history_store.max_length)
        self.features = np.zeros(
            (len(self.records), sequence_length, 2), dtype=np.float32
        )
        self.mask = np.zeros((len(self.records), sequence_length), dtype=np.bool_)
        self.history_count = np.zeros(len(self.records), dtype=np.int64)
        for row_index, video_id in enumerate(self.records["video_id"].astype(str)):
            history_row = history_lookup.get(video_id)
            if history_row is None:
                raise KeyError(f"Missing history sequence for {video_id}")
            start = int(history_store.offsets[history_row])
            end = int(history_store.offsets[history_row + 1])
            count = end - start
            self.history_count[row_index] = count
            if count:
                self.features[row_index, :count] = history_store.features[start:end]
                self.mask[row_index, :count] = True
        self.labels = self.records["abnormal_score"].to_numpy(np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.mask[index]),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


def load_task(reference_dir, target):
    records = pd.read_csv(
        reference_dir / "task_records" / f"{target}.csv",
        dtype={"hospital_id": str, "video_id": str},
    )
    if set(records["split"]) != {"train", "val", "test"}:
        raise ValueError(f"Unexpected split values for {target}")
    if records["video_id"].duplicated().any():
        raise ValueError(f"Duplicate video IDs for {target}")
    history = HistoryFeatureStore.load(
        reference_dir / "history_records" / f"{target}.npz"
    )
    if set(records["video_id"].astype(str)) != set(history.video_ids.astype(str)):
        raise ValueError(f"Task/history video sets differ for {target}")
    return records, history
