"""Reuse Exp6 frame alignment and weights, replacing pixels with frozen CLS."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from study.common.video_loss import DistinctLabViewBatchSampler
from study.exp6_face_pair_lab_delta.data import PairedFrameDataset

from . import config


class FeaturePairs(PairedFrameDataset):
    def __init__(self, index, records, views, cache_dir=config.CACHE, feature_count=384):
        super().__init__(index, records, views=views, index_views=len(views) > 1)
        self.features = np.load(cache_dir / "cls_features.npy", mmap_mode="r")
        if list(self.features.shape) != [len(index.starts), 5, feature_count]:
            raise ValueError("Frozen feature dimensions differ from the Exp6 frame index")

    def __getitem__(self, sample):
        frame, view = divmod(sample, len(self.views))
        pair = int(self.frame_pair_rows[frame])
        difference = (self.features[self.second_indices[frame], view]
                      - self.features[self.first_indices[frame], view])
        return (torch.from_numpy(difference.copy()), torch.tensor(self.labels[pair]),
                torch.tensor(frame), torch.tensor(self.pair_weights[pair]))


def loader(index, records, train, cache_dir=config.CACHE):
    dataset = FeaturePairs(index, records, config.VIEWS if train else ("original",), cache_dir)
    if train:
        return dataset, DataLoader(dataset, batch_sampler=DistinctLabViewBatchSampler(dataset, config.FRAME_PAIR_BATCH_SIZE),
                                   num_workers=0, pin_memory=torch.cuda.is_available())
    return dataset, DataLoader(dataset, batch_size=480, shuffle=False, num_workers=0,
                               pin_memory=torch.cuda.is_available())
