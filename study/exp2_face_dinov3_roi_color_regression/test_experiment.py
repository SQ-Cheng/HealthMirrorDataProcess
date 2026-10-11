"""Focused alignment, batch-coverage, normalization and head checks."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from .data import lab_batches, fit_color_scaler, pooled_prediction, load_features
from .models import FusionHead


class FusionTests(unittest.TestCase):
    def test_frame_and_view_alignment_after_filtering(self):
        index = SimpleNamespace(starts=np.zeros(40), frame_range=lambda video: (0, 20) if video == "a" else (20, 40))
        identity = np.arange(40)[:, None, None] * 100 + np.arange(5)[None, :, None]
        dino = np.broadcast_to(identity, (40, 5, 384)).copy().astype(np.float32)
        color = np.broadcast_to(identity[..., None], (40, 5, 4, 41)).copy().astype(np.float32)
        valid = np.ones((40, 5, 4), bool)
        valid[[2, 5, 29], 3, 1] = False
        records = pd.DataFrame({"video_id": ["b", "a"]})
        with patch("study.exp2_face_dinov3_roi_color_regression.data.np.load", side_effect=[dino, color, valid]):
            fused, ptr, frames = load_features(index, records)
        np.testing.assert_array_equal(ptr, [0, 19, 37])
        np.testing.assert_array_equal(frames[:19], np.delete(np.arange(20, 40), 9))
        np.testing.assert_array_equal(fused[:, :, 0], fused[:, :, 384])
        np.testing.assert_array_equal(fused[:, :, 0], frames[:, None] * 100 + np.arange(5))

    def test_parameter_count_and_gradients(self):
        model = FusionHead()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 17665)
        self.assertEqual(sum(p.numel() for p in model.buffers()), 328)
        model(torch.randn(12, 548)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_sampler_exactly_once_and_distinct_events(self):
        records = pd.DataFrame({"clinical_event_id": [f"event{i // 2}" for i in range(30)]})
        counts = np.arange(30) % 11 + 10
        ptr = np.concatenate(([0], np.cumsum(counts)))
        rows = np.repeat(np.arange(30), counts)
        batches = list(lab_batches(records, ptr))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(ptr[-1] * 5))
        for batch in batches:
            selected = np.unique(rows[batch // 5] * 5 + batch % 5)
            self.assertLessEqual(len(selected), 12)
            self.assertEqual(records.iloc[selected // 5].clinical_event_id.nunique(), len(selected))
            for group in selected:
                video, view = divmod(group, 5)
                expected = np.arange(ptr[video], ptr[video + 1]) * 5 + view
                np.testing.assert_array_equal(np.sort(batch[(rows[batch // 5] == video) & (batch % 5 == view)]), expected)

    def test_scaling_equal_video_mass_no_held_out_data(self):
        features = np.zeros((36, 5, 548), np.float32)
        ptr = np.array([0, 10, 30, 36])
        features[:10, :, 384:] = 2
        features[10:30, :, 384:] = 6
        features[30:, :, 384:] = 1e8
        scaler = fit_color_scaler(features, ptr, [0, 1])
        np.testing.assert_allclose(scaler.mean_, 4)
        np.testing.assert_allclose(scaler.scale_, 2)
        np.testing.assert_array_equal(pooled_prediction(np.arange(6.), [0, 2, 6]), [.5, 3.5])


if __name__ == "__main__":
    unittest.main()
