"""Streaming paired-frame dataset with no decoded-image cache on disk."""

from collections import OrderedDict
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from .config import DECODE_CACHE_FRAMES, MAX_OPEN_FILES_PER_WORKER, SOURCE_IMAGE_SIZE


class PairedFrameDataset(Dataset):
    def __init__(self, frame_index, records, views=("original",), expand_views=False):
        self.index = frame_index
        self.records = records.reset_index(drop=True).copy()
        self.views = tuple(views)
        self.expand_views = bool(expand_views)
        post_indices, pre_indices, video_rows = [], [], []
        for video_row, row in enumerate(self.records.itertuples(index=False)):
            post_start, post_end = frame_index.frame_range(row.video_id)
            pre_start, pre_end = frame_index.frame_range(row.pre_video_id)
            post = np.arange(post_start, post_end, dtype=np.int64)
            pre = np.arange(pre_start, pre_end, dtype=np.int64)
            if len(post) != len(pre):
                raise ValueError(f"Pre/post frame-count mismatch for {row.video_id}")
            offset = int.from_bytes(
                hashlib.sha256(str(row.video_id).encode()).digest()[:2], "little"
            ) % len(pre)
            post_indices.append(post)
            pre_indices.append(np.roll(pre, offset))
            video_rows.append(np.full(len(post), video_row, dtype=np.int32))
        self.post_indices = np.concatenate(post_indices)
        self.pre_indices = np.concatenate(pre_indices)
        self.frame_video_rows = np.concatenate(video_rows)
        self.labels = self.records.recovery_score.to_numpy(np.float32)
        counts = self.records.groupby("hospital_id").video_id.transform("size")
        weights = 1.0 / counts.to_numpy(np.float32)
        self.video_weights = weights / weights.mean()
        self._handles, self._decoded = OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.post_indices)

    @property
    def model_input_count(self):
        return len(self) * (len(self.views) if self.expand_views else 1)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"], state["_decoded"] = OrderedDict(), OrderedDict()
        return state

    def _handle(self, path):
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = open(path, "rb")
        self._handles[path] = handle
        while len(self._handles) > MAX_OPEN_FILES_PER_WORKER:
            _, old = self._handles.popitem(last=False); old.close()
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
        video_row = int(self.frame_video_rows[frame_row])
        view_codes = (
            torch.arange(len(self.views), dtype=torch.uint8)
            if self.expand_views else torch.tensor(0, dtype=torch.uint8)
        )
        return (
            self._decode(int(self.pre_indices[frame_row])),
            self._decode(int(self.post_indices[frame_row])),
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


class SingleFrameDataset(PairedFrameDataset):
    """Use one side of the exact paired-frame index for controlled ablations."""

    def __init__(self, frame_index, records, mode, views=("original",), expand_views=False):
        if mode not in {"pre_only", "post_only"}:
            raise ValueError(f"Unsupported single-frame mode: {mode}")
        super().__init__(frame_index, records, views, expand_views)
        self.mode = mode

    def __getitem__(self, frame_row):
        video_row = int(self.frame_video_rows[frame_row])
        view_codes = (
            torch.arange(len(self.views), dtype=torch.uint8)
            if self.expand_views else torch.tensor(0, dtype=torch.uint8)
        )
        indices = self.pre_indices if self.mode == "pre_only" else self.post_indices
        return (
            self._decode(int(indices[frame_row])),
            torch.tensor(self.labels[video_row]),
            torch.tensor(frame_row, dtype=torch.long),
            view_codes,
            torch.tensor(self.video_weights[video_row]),
        )
