"""Focused complete-video coverage, objective, and gradient tests."""

import math
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .video_loss import VideoBCELoss, VideoSmoothL1Loss, VideoViewBatchSampler, DistinctLabViewBatchSampler


class VideoLossTests(unittest.TestCase):
    def test_distinct_labs_separate_views_without_dropping_data(self):
        dataset = SimpleNamespace(
            expand_all_views=False,
            video_records=pd.DataFrame({"video_id": range(60), "hospital_id": ["one_patient"] * 60,
                                        "clinical_event_id": [f"lab_{i // 2}" for i in range(60)]}),
            frame_video_rows=np.repeat(np.arange(60), 20), views=tuple(range(5)),
        )
        sampler = DistinctLabViewBatchSampler(dataset, 240)
        torch.manual_seed(9)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(len(batches), math.ceil(60 * 100 / 240))
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(60 * 100)))
        for batch in batches:
            self.assertEqual(len(batch), 240)
            groups = np.asarray(batch).reshape(12, 20)
            videos = groups[:, 0] // 100
            self.assertEqual(len(set(videos)), 12)
            self.assertEqual(dataset.video_records.iloc[videos].clinical_event_id.nunique(), 12)
            for group in groups:
                self.assertEqual(len(set(group % 5)), 1)
                self.assertEqual(len(set(group // 100)), 1)
                self.assertEqual(sorted((group // 5) % 20), list(range(20)))
        torch.manual_seed(9)
        self.assertEqual(batches, list(sampler))

    def test_distinct_labs_retains_partial_tail(self):
        dataset = SimpleNamespace(
            expand_all_views=False,
            video_records=pd.DataFrame({"clinical_event_id": [f"lab_{i}" for i in range(31)]}),
            frame_video_rows=np.repeat(np.arange(31), 20), views=tuple(range(5)),
        )
        batches = list(DistinctLabViewBatchSampler(dataset, 240))
        self.assertTrue(all(len(batch) == 240 for batch in batches[:-1]))
        self.assertEqual(len(batches[-1]), 220)
        self.assertEqual(sum(map(len, batches)), 3100)

    def test_sampler_preserves_every_frame_and_view(self):
        dataset = SimpleNamespace(
            expand_all_views=False, video_records=pd.DataFrame({"video_id": range(17)}),
            frame_video_rows=np.repeat(np.arange(17), 20), views=tuple(range(5)),
        )
        sampler = VideoViewBatchSampler(dataset, 240, True)
        torch.manual_seed(12)
        batches = list(sampler)
        self.assertEqual(len(batches), math.ceil(17 * 100 / 240))
        all_indices = [i for batch in batches for i in batch]
        self.assertEqual(sorted(all_indices), list(range(17 * 100)))
        for batch in batches:
            self.assertLessEqual(len(batch), 240)
            groups = np.array(batch).reshape(-1, 20)
            for group in groups:
                self.assertEqual(len(set(group // 100)), 1)
                self.assertEqual(len(set(group % 5)), 1)
                self.assertEqual(sorted((group // 5) % 20), list(range(20)))

    def test_bce_matches_weighted_mean_probability_and_has_gradients(self):
        logits = torch.linspace(-3, 3, 40, requires_grad=True)
        labels = torch.tensor([0.] * 20 + [1.] * 20)
        probability = logits.sigmoid().reshape(2, 20).mean(1)
        expected = -(2 * labels[::20] * probability.log()
                     + (1 - labels[::20]) * (1 - probability).log())
        loss = VideoBCELoss(2.)(logits, labels)
        torch.testing.assert_close(loss, expected)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(logits.grad.ne(0).all())

    def test_extreme_logits_are_finite_without_probability_clipping(self):
        logits = torch.tensor([100.] * 20 + [-100.] * 20, requires_grad=True)
        labels = torch.tensor([0.] * 20 + [1.] * 20)
        loss = VideoBCELoss(2.)(logits, labels)
        self.assertTrue(torch.isfinite(loss).all())
        loss.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(logits.grad.ne(0).all())

    def test_regression_pools_before_smooth_l1(self):
        prediction = torch.tensor([-1., 1.] * 10, requires_grad=True)
        truth = torch.zeros(20)
        loss = VideoSmoothL1Loss()(prediction, truth)
        self.assertEqual(loss.item(), 0.)
        self.assertGreater(F.smooth_l1_loss(prediction, truth, beta=.5).item(), 0)
        prediction = torch.arange(40, dtype=torch.float32, requires_grad=True)
        loss = VideoSmoothL1Loss()(prediction, torch.zeros(40))
        loss.mean().backward()
        self.assertTrue(prediction.grad.ne(0).all())


if __name__ == "__main__":
    unittest.main()
