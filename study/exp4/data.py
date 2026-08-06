"""Streaming frame datasets for recovery regression."""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .config import (
    DECODE_CACHE_FRAMES,
    MAX_OPEN_FILES_PER_WORKER,
    SOURCE_IMAGE_SIZE,
)


class RecoveryFrameDataset(Dataset):
    def __init__(self, frame_index, records, views=("original",), expand_views=False):
        self.index = frame_index
        self.records = records.reset_index(drop=True).copy()
        self.views = tuple(views)
        self.expand_views = bool(expand_views)
        frame_indices, frame_video_rows = [], []
        for video_row, row in enumerate(self.records.itertuples(index=False)):
            start, end = frame_index.frame_range(row.video_id)
            frame_indices.append(np.arange(start, end, dtype=np.int64))
            frame_video_rows.append(np.full(end - start, video_row, dtype=np.int32))
        self.frame_indices = np.concatenate(frame_indices)
        self.frame_video_rows = np.concatenate(frame_video_rows)
        self.labels = self.records["recovery_score"].to_numpy(np.float32)
        patient_video_counts = self.records.groupby("hospital_id")["video_id"].transform("size")
        weights = 1.0 / patient_video_counts.to_numpy(np.float32)
        self.video_weights = weights / weights.mean()
        self._handles = OrderedDict()
        self._decoded = OrderedDict()

    def __len__(self):
        return len(self.frame_indices)

    @property
    def model_input_count(self):
        return len(self.frame_indices) * (len(self.views) if self.expand_views else 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        state["_decoded"] = OrderedDict()
        return state

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > MAX_OPEN_FILES_PER_WORKER:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def _decode(self, global_index):
        cached = self._decoded.pop(global_index, None)
        if cached is not None:
            self._decoded[global_index] = cached
            return cached
        video_position = int(np.searchsorted(self.index.video_ptr, global_index, side="right") - 1)
        path = str(self.index.video_paths[video_position])
        start, end = int(self.index.starts[global_index]), int(self.index.ends[global_index])
        handle = self._handle(path); handle.seek(start)
        encoded = torch.frombuffer(bytearray(handle.read(end - start)), dtype=torch.uint8)
        image = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
        if tuple(image.shape) != (3, SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
            raise RuntimeError(f"Unexpected frame shape {tuple(image.shape)} in {path}")
        self._decoded[global_index] = image
        while len(self._decoded) > DECODE_CACHE_FRAMES:
            self._decoded.popitem(last=False)
        return image

    def __getitem__(self, frame_row):
        global_index = int(self.frame_indices[frame_row])
        video_row = int(self.frame_video_rows[frame_row])
        view_codes = (
            torch.arange(len(self.views), dtype=torch.uint8)
            if self.expand_views
            else torch.tensor(0, dtype=torch.uint8)
        )
        return (
            self._decode(global_index),
            torch.tensor(self.labels[video_row]),
            torch.tensor(frame_row, dtype=torch.long),
            view_codes,
            torch.tensor(self.video_weights[video_row]),
        )

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear(); self._decoded.clear()

    def __del__(self):
        self.close()
