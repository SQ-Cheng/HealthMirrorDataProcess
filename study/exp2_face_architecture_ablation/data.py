"""Reuse the same clinical measurement groups, selected frames, and views."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset

from .features import CACHE
from .config import VIEWS


class FeatureFrameDataset(Dataset):
    def __init__(self, index, records, views, architecture, family):
        self.index = index
        self.video_records = records.reset_index(drop=True)
        self.views, self.expand_all_views = tuple(views), False
        self.frame_indices = np.concatenate([np.arange(*index.frame_range(video))
                                             for video in records.video_id])
        self.frame_video_rows = np.repeat(np.arange(len(records)), 20)
        self.frame_count = len(self.frame_indices)
        self.labels_by_video = records["binary_label" if family == "classification" else "robust_scaled_raw_value"].to_numpy(np.float32)
        name = "histogram" if architecture == "color_histogram_mlp" else "statistics"
        self.features = np.load(CACHE / f"{name}.npy", mmap_mode="r")

    def __len__(self):
        return self.frame_count * len(self.views)

    def __getitem__(self, sample):
        frame, view = divmod(sample, len(self.views))
        video = self.frame_video_rows[frame]
        features = self.features[self.frame_indices[frame], view].copy()
        return torch.from_numpy(features), torch.tensor(self.labels_by_video[video]), torch.tensor(frame), torch.tensor(view)


def loader(index, records, architecture, family, train):
    views = VIEWS if train else ("original",)
    if architecture == "small_cnn":
        dataset = AllFramesDataset(index, records, views=views, expand_all_views=False, interpolation="bicubic")
        dataset.labels_by_video = records["binary_label" if family == "classification" else "robust_scaled_raw_value"].to_numpy(np.float32)
        dataset.decode_cache_frames = 20
        workers = 6 if train else 2
    else:
        dataset = FeatureFrameDataset(index, records, views, architecture, family)
        workers = 0
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    options = {"batch_sampler": sampler, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=4)
    return dataset, DataLoader(dataset, **options)


def fit_feature_scaler(model, index, records, architecture):
    if architecture == "small_cnn":
        return
    name = "histogram" if architecture == "color_histogram_mlp" else "statistics"
    features = np.load(CACHE / f"{name}.npy", mmap_mode="r")
    indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.loc[records.split.eq("train"), "video_id"]])
    training = features[indices].reshape(-1, features.shape[-1]).astype(np.float64)
    model.feature_mean.copy_(torch.from_numpy(training.mean(0).astype(np.float32)))
    model.feature_std.copy_(torch.from_numpy(np.maximum(training.std(0), 1e-4).astype(np.float32)))
