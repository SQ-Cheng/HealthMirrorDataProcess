"""Focused data/contrastive invariants for the two 30/40 ablations."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from .density_weighting import make_density_weights
from .label_simclr import LabelPairBatchSampler, contrastive_loss
from .train import _training_loss


class DensityWeightsTest(unittest.TestCase):
    def test_sparse_range_receives_more_weight(self):
        values = np.r_[np.zeros(100), np.full(4, 3.0)]
        records = pd.DataFrame({
            "split": ["train"] * len(values),
            "robust_scaled_raw_value": values,
        })
        weights, audit = make_density_weights(records)
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertEqual(sum(audit["training_videos_per_bin"]), len(values))

    def test_validation_record_cannot_fit_density(self):
        records = pd.DataFrame({
            "split": ["train"] * 12 + ["val"],
            "robust_scaled_raw_value": np.arange(13),
        })
        with self.assertRaises(ValueError):
            make_density_weights(records)

    def test_five_views_inherit_their_video_weight(self):
        losses = torch.tensor([1.0] * 5 + [3.0] * 5).unsqueeze(1)
        weights = torch.tensor([1.0, 3.0])
        value = _training_loss(
            losses, weights, np.array([0, 1]), torch.tensor([0, 1]),
            torch.zeros((2, 5), dtype=torch.uint8), torch.device("cpu"),
        )
        self.assertAlmostEqual(value.item(), 2.5)


class ContrastivePairsTest(unittest.TestCase):
    def test_same_patient_different_event_can_be_positive(self):
        records = pd.DataFrame({
            "split": ["train"] * 3,
            "robust_scaled_raw_value": [0.0, 0.1, 1.5],
            "hospital_id": ["A", "A", "B"],
            "source_sample_id": ["event1", "event2", "event3"],
        })
        dataset = SimpleNamespace(
            video_records=records,
            frame_video_rows=np.repeat(np.arange(3), 20),
            views=("original",),
        )
        sampler = LabelPairBatchSampler(dataset, anchors_per_batch=2)
        self.assertEqual(sampler._partner(0), (1, "same_patient_new_event"))
        self.assertEqual(
            sampler.audit()["videos_with_same_patient_different_event_close_partner"], 2
        )
        self.assertTrue(all(0 <= index < 60 for batch in sampler for index in batch))

    def test_ambiguous_values_are_not_negatives(self):
        values = torch.tensor([0.0, 0.0, 0.1, 0.1, 1.0, 1.0])
        video_rows = torch.tensor([0, 0, 1, 1, 2, 2])
        embeddings = torch.randn((6, 8))
        loss, audit = contrastive_loss(embeddings, values, video_rows)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(audit["positive_pairs"], 14)
        self.assertEqual(audit["far_negative_pairs"], 16)
        self.assertEqual(audit["valid_anchors"], 6)


if __name__ == "__main__":
    unittest.main()
