"""Focused tests for the controlled regularization ablations."""

import unittest

import torch
from torch import nn

from .train import _freeze_batchnorm_running_stats


class BatchNormStatisticsTest(unittest.TestCase):
    def test_running_statistics_stop_but_affine_parameters_remain_trainable(self):
        model = nn.Sequential(nn.BatchNorm2d(2), nn.Dropout2d(p=0.25))
        model.train()
        batchnorm = model[0]
        before_mean = batchnorm.running_mean.clone()
        before_count = batchnorm.num_batches_tracked.clone()
        _freeze_batchnorm_running_stats(model)

        self.assertFalse(batchnorm.training)
        self.assertTrue(model[1].training)
        self.assertTrue(batchnorm.weight.requires_grad)
        self.assertTrue(batchnorm.bias.requires_grad)
        model(torch.ones(4, 2, 3, 3)).sum().backward()
        self.assertTrue(torch.equal(batchnorm.running_mean, before_mean))
        self.assertTrue(torch.equal(batchnorm.num_batches_tracked, before_count))
        self.assertIsNotNone(batchnorm.weight.grad)


if __name__ == "__main__":
    unittest.main()
