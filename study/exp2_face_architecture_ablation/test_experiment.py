"""CPU smoke tests for colors, model sizes, and the unchanged view objective."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import pandas as pd

from study.common.video_loss import VideoBCELoss, VideoSmoothL1Loss
from .features import color_features
from .models import build_model
from .data import FeatureFrameDataset, fit_feature_scaler


class ArchitectureTests(unittest.TestCase):
    def test_feature_scaler_uses_training_only(self):
        records = pd.DataFrame({"video_id": ["a", "b", "c"], "split": ["train", "val", "test"]})
        ranges = {"a": (0, 20), "b": (20, 40), "c": (40, 60)}
        index = SimpleNamespace(frame_range=lambda video: ranges[video])
        values = np.ones((60, 5, 43), np.float32)
        values[20:] = 100
        model = build_model("color_statistics_mlp")
        with patch("study.exp2_face_architecture_ablation.data.np.load", return_value=values):
            fit_feature_scaler(model, index, records, "color_statistics_mlp")
        torch.testing.assert_close(model.feature_mean, torch.ones(43))
        torch.testing.assert_close(model.feature_std, torch.full((43,), 1e-4))

    def test_feature_rows_keep_global_frame_and_view_identity(self):
        records = pd.DataFrame({"video_id": ["a"], "binary_label": [1], "robust_scaled_raw_value": [7.]})
        index = SimpleNamespace(frame_range=lambda _: (20, 40))
        values = np.zeros((40, 5, 43), np.float32)
        values[20, 1] = 201
        with patch("study.exp2_face_architecture_ablation.data.np.load", return_value=values):
            dataset = FeatureFrameDataset(index, records, tuple(range(5)), "color_statistics_mlp", "classification")
            features, label, frame, view = dataset[1]
        self.assertEqual(len(dataset), 100)
        self.assertTrue(features.eq(201).all())
        self.assertEqual((label.item(), frame.item(), view.item()), (1, 0, 1))

    def setUp(self):
        torch.set_num_threads(1)
        cv2.setNumThreads(1)

    def test_parameter_counts(self):
        expected = {"color_histogram_mlp": 11585, "color_statistics_mlp": 5121, "small_cnn": 97025}
        for architecture, count in expected.items():
            model = build_model(architecture)
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)

    def test_constant_and_circular_hue_features(self):
        histogram, statistics = color_features(np.full((24, 24, 3), .5, np.float32))
        np.testing.assert_allclose(histogram.reshape(9, 16).sum(1), 1.)
        np.testing.assert_allclose(statistics[:3], .5, atol=1e-6)
        np.testing.assert_allclose(statistics[25:28], 0., atol=1e-6)
        hsv = np.ones((2, 24, 3), np.float32)
        hsv[0, :, 0] = 359
        hsv[1, :, 0] = 1
        _, statistics = color_features(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        self.assertGreater(statistics[26], .99)
        self.assertLess(abs(statistics[25]), .02)
        self.assertTrue(np.isfinite(statistics).all())

    def test_view_losses_backpropagate_through_each_model(self):
        for architecture, width in (("color_histogram_mlp", 144), ("color_statistics_mlp", 43), ("small_cnn", 0)):
            model = build_model(architecture)
            inputs = torch.randn(20, width) if width else torch.randn(20, 3, 64, 64)
            for criterion in (VideoBCELoss(.2), VideoSmoothL1Loss()):
                model.zero_grad(set_to_none=True)
                output = model(inputs).squeeze(1)
                loss = criterion(output, torch.ones(20)).mean()
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)


if __name__ == "__main__":
    unittest.main()
