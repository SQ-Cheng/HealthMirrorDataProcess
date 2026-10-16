"""Aligned read-only feature banks; identical batches for all six structures."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_architecture_ablation.data import FeatureFrameDataset
from . import config


class PairedFeatures(FeatureFrameDataset):
    def __init__(self, index, records, views, architecture):
        self.index = index
        self.video_records = records.reset_index(drop=True)
        self.views, self.expand_all_views = tuple(views), False
        self.frame_indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.video_id])
        self.frame_video_rows = np.repeat(np.arange(len(records)), 20)
        self.frame_count = len(self.frame_indices)
        self.labels_by_video = records.robust_scaled_raw_value.to_numpy(np.float32)
        self.banks = [np.load(config.CACHE / f"{branch}.npy", mmap_mode="r") for branch in config.branches_for(architecture)]
        for bank, branch in zip(self.banks, config.branches_for(architecture)):
            if bank.shape != (len(index.starts), 5, config.DIMENSIONS[branch]):
                raise RuntimeError("Paired cache/index shape mismatch")

    def __getitems__(self, samples):
        frames, views = np.divmod(np.asarray(samples), len(self.views))
        source = self.frame_indices[frames]
        values = torch.from_numpy(np.concatenate([bank[source, views] for bank in self.banks], axis=-1))
        labels = torch.from_numpy(self.labels_by_video[self.frame_video_rows[frames]])
        return list(zip(values, labels, torch.from_numpy(frames), torch.from_numpy(views)))

    def __getitem__(self, sample):
        return self.__getitems__([sample])[0]


def loader(index, records, architecture, family, train):
    if family != "regression":
        raise ValueError(family)
    dataset = PairedFeatures(index, records, config.VIEWS if train else ("original",), architecture)
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
