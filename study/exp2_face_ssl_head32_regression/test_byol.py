"""Stop-gradient, EMA, full frame/view coverage and diverse anchor batch tests."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

from .byol import BYOL, DiverseVideoBatchSampler, RankBatchSampler, bootstrap_loss, schedules
from . import config


class BYOLTests(unittest.TestCase):
    def test_normalized_loss_and_stop_gradient(self):
        prediction = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        target = torch.tensor([[1., 0.], [0., -1.]], requires_grad=True)
        loss = bootstrap_loss(prediction, target)
        torch.testing.assert_close(loss, torch.tensor([0., 4.]))
        loss.sum().backward(); self.assertIsNone(target.grad)

    def test_ema_updates_parameters_but_not_teacher_running_buffers(self):
        model = BYOL.__new__(BYOL); nn.Module.__init__(model)
        model.online_features = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model.online_projector = nn.Linear(2, 2)
        model.target_features = copy.deepcopy(model.online_features)
        model.target_projector = copy.deepcopy(model.online_projector)
        with torch.no_grad():
            for module in (model.online_features, model.online_projector):
                for parameter in module.parameters(): parameter.fill_(2)
            for module in (model.target_features, model.target_projector):
                for parameter in module.parameters(): parameter.zero_()
            model.target_features[1].running_mean.fill_(3)
        model.update_target(.9)
        torch.testing.assert_close(model.target_projector.weight, torch.full_like(model.target_projector.weight, .2))
        torch.testing.assert_close(model.target_features[1].running_mean, torch.tensor([3., 3.]))

    def test_every_frame_view_once_and_video_diversity(self):
        counts = [3] * 12
        sampler = DiverseVideoBatchSampler(counts, 4)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        flat = [index for batch in batches for index, epoch in batch]
        self.assertEqual(sorted(flat), list(range(sum(counts) * 5)))
        for batch in batches:
            self.assertEqual(len({index // 15 for index, epoch in batch}), len(batch))

    def test_long_video_tail_fills_batches_without_duplicate_indices(self):
        sampler = DiverseVideoBatchSampler([100, 1, 1], 64)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        flat = [index for batch in batches for index, epoch in batch]
        self.assertEqual(sorted(flat), list(range(510)))
        self.assertTrue(all(len(batch) == 64 for batch in batches[:-1]))
        sampler.epoch = 1
        self.assertNotEqual(batches, list(sampler))

    def test_learning_rate_and_ema_schedules(self):
        self.assertGreater(schedules(0, 100, 10)[0], 0)
        self.assertEqual(schedules(100, 100, 10)[0], 0)
        self.assertAlmostEqual(schedules(0, 100, 10)[1], .996)
        self.assertEqual(schedules(100, 100, 10)[1], 1)

    def test_distributed_tail_preserves_real_anchors_and_masks_padding(self):
        base = DiverseVideoBatchSampler([1], 4)
        batches = [list(RankBatchSampler(base, rank, 4)) for rank in range(4)]
        real = []
        for step in range(len(base)):
            for rank in range(4):
                for index, epoch, valid in batches[rank][step]:
                    if valid: real.append(index)
        self.assertEqual(sorted(real), list(range(5)))
        self.assertTrue(all(len(rank_batches) == len(base) for rank_batches in batches))

    def test_online_predictor_gradients_teacher_has_no_gradients(self):
        torch.set_num_threads(1)
        model = BYOL()
        loss, features = model(torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(any(p.grad is not None for p in model.predictor.parameters()))
        self.assertFalse(any(p.grad is not None for p in model.target_features.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.target_projector.parameters()))

    def test_all_five_views_get_distinct_partners(self):
        for epoch in range(4):
            for frame in range(20):
                for anchor in range(5):
                    partner = (anchor + 1 + (frame * 2027 + epoch * 1009) % 4) % len(config.VIEWS)
                    self.assertNotEqual(anchor, partner)
                    self.assertIn(partner, range(5))


if __name__ == "__main__":
    unittest.main()
