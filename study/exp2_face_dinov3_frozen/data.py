"""Same twelve-measurement batches, now backed by frozen DINOv3 CLS vectors."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_architecture_ablation.data import FeatureFrameDataset

from . import config
from .features import CACHE


class DinoFeatureDataset(FeatureFrameDataset):
    def __init__(self, index, records, views, family, cache_dir=CACHE):
        self.index = index
        self.video_records = records.reset_index(drop=True)
        self.views, self.expand_all_views = tuple(views), False
        self.frame_indices = np.concatenate([np.arange(*index.frame_range(video)) for video in records.video_id])
        self.frame_video_rows = np.repeat(np.arange(len(records)), 20)
        self.frame_count = len(self.frame_indices)
        self.labels_by_video = records["binary_label" if family == "classification" else "robust_scaled_raw_value"].to_numpy(np.float32)
        self.features = np.load(cache_dir / "cls_features.npy", mmap_mode="r")


def loader(index, records, architecture, family, train):
    dataset = DinoFeatureDataset(index, records, config.VIEWS if train else ("original",), family)
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())


def build_head(_):
    from .backbone import Head
    return Head()


def no_feature_fitting(*_):
    """The pretrained final CLS LayerNorm is already part of the frozen encoder."""
