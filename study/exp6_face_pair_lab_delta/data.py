"""Streaming paired-frame input without decoded-frame persistence."""

from collections import OrderedDict, deque

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import ImageReadMode, decode_jpeg

from .config import (
    DECODE_CACHE_FRAMES,
    MAX_OPEN_FILES_PER_WORKER,
    SOURCE_IMAGE_SIZE,
)


class PairedFrameDataset(Dataset):
    def __init__(self, frame_index, records, views=("original",), expand_views=False):
        self.index = frame_index
        self.records = records.reset_index(drop=True).copy()
        self.views = tuple(views)
        self.expand_views = bool(expand_views)
        first_indices, second_indices, pair_rows = [], [], []
        for pair_row, row in enumerate(self.records.itertuples(index=False)):
            first_start, first_end = frame_index.frame_range(row.first_video_id)
            second_start, second_end = frame_index.frame_range(row.second_video_id)
            first = np.arange(first_start, first_end, dtype=np.int64)
            second = np.arange(second_start, second_end, dtype=np.int64)
            if len(first) != 20 or len(second) != 20:
                raise ValueError(
                    f"Expected 20+20 frames for pair {row.pair_id}, "
                    f"found {len(first)}+{len(second)}"
                )
            first_indices.append(first)
            second_indices.append(second)
            pair_rows.append(np.full(20, pair_row, dtype=np.int32))
        self.first_indices = np.concatenate(first_indices)
        self.second_indices = np.concatenate(second_indices)
        self.frame_pair_rows = np.concatenate(pair_rows)
        self.labels = self.records["scaled_delta"].to_numpy(np.float32)
        patient_pair_counts = self.records.groupby("hospital_id").pair_id.transform("size")
        weights = 1.0 / patient_pair_counts.to_numpy(np.float32)
        self.pair_weights = weights / weights.mean()
        self._handles = OrderedDict()
        self._decoded = OrderedDict()

    def __len__(self):
        return len(self.first_indices)

    @property
    def model_input_count(self):
        return len(self) * (len(self.views) if self.expand_views else 1)

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
        video_position = int(
            np.searchsorted(self.index.video_ptr, global_index, side="right") - 1
        )
        path = str(self.index.video_paths[video_position])
        start = int(self.index.starts[global_index])
        end = int(self.index.ends[global_index])
        handle = self._handle(path)
        handle.seek(start)
        encoded = torch.frombuffer(
            bytearray(handle.read(end - start)), dtype=torch.uint8
        )
        image = decode_jpeg(encoded, mode=ImageReadMode.RGB, device="cpu")
        if tuple(image.shape) != (3, SOURCE_IMAGE_SIZE, SOURCE_IMAGE_SIZE):
            raise RuntimeError(f"Unexpected frame shape {tuple(image.shape)} in {path}")
        self._decoded[global_index] = image
        while len(self._decoded) > DECODE_CACHE_FRAMES:
            self._decoded.popitem(last=False)
        return image

    def __getitem__(self, frame_row):
        pair_row = int(self.frame_pair_rows[frame_row])
        view_codes = (
            torch.arange(len(self.views), dtype=torch.uint8)
            if self.expand_views
            else torch.tensor(0, dtype=torch.uint8)
        )
        return (
            self._decode(int(self.first_indices[frame_row])),
            self._decode(int(self.second_indices[frame_row])),
            torch.tensor(self.labels[pair_row]),
            torch.tensor(frame_row, dtype=torch.long),
            view_codes,
            torch.tensor(self.pair_weights[pair_row]),
        )

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._decoded.clear()

    def __del__(self):
        self.close()


class ChunkShuffleSampler(Sampler):
    """Shuffle source pairs while keeping their 20 reads locally coherent."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        pair_count = len(self.dataset.records)
        order = torch.randperm(pair_count).tolist()
        for pair_row in order:
            start = pair_row * 20
            frame_order = torch.randperm(20).tolist()
            for offset in frame_order:
                yield start + offset


class PatientDiversePairSampler(Sampler):
    """Use every frame pair once while mixing patients in each source batch."""

    def __init__(self, dataset, source_batch_size, frames_per_group=2):
        if (frames_per_group < 1 or 20 % frames_per_group
                or source_batch_size < frames_per_group
                or source_batch_size % frames_per_group):
            raise ValueError("Frame groups must divide 20 and the source batch size")
        self.dataset = dataset
        self.groups_per_batch = source_batch_size // frames_per_group
        self.groups = []
        for pair_row, patient_id in enumerate(dataset.records.hospital_id.astype(str)):
            start = pair_row * 20
            for offset in range(0, 20, frames_per_group):
                self.groups.append((patient_id, range(start + offset, start + offset + frames_per_group)))

    def __iter__(self):
        pending = deque(torch.randperm(len(self.groups)).tolist())
        while pending:
            chosen, patients = [], set()
            for _ in range(len(pending)):
                if len(chosen) == self.groups_per_batch:
                    break
                group_index = pending.popleft()
                patient_id = self.groups[group_index][0]
                if patient_id in patients:
                    pending.append(group_index)
                else:
                    chosen.append(group_index)
                    patients.add(patient_id)
            while pending and len(chosen) < self.groups_per_batch:
                chosen.append(pending.popleft())
            for group_index in chosen:
                yield from self.groups[group_index][1]

    def __len__(self):
        return len(self.dataset)
