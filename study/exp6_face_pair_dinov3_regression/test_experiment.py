"""CPU checks for cached differences, patient weights and complete pair batches."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.common.video_loss import DistinctLabViewBatchSampler
from study.exp6_face_pair_lab_delta.data import PairedFrameDataset
from study.exp6_face_pair_lab_delta.train import _weighted_loss
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from .data import FeaturePairs
from .models import FrozenDeltaPredictor, head_parameters
from .train import train_task


def synthetic_data(path):
    index = FrameOffsetIndex(np.array(["early", "late"]), np.array(["early.mkv", "late.mkv"]),
                             np.array([0, 20, 40]), np.arange(40), np.arange(40) + 1,
                             np.tile(np.arange(20), 2), np.array(["ffv1", "ffv1"]))
    features = np.zeros((40, 5, 384), np.float32)
    for view in range(5):
        features[:20, view] = 2 * view
        features[20:, view] = 3 * view + np.arange(20)[:, None]
    np.save(path / "cls_features.npy", features)
    rows = []
    for i in range(16):
        split = "train" if i < 12 else "val" if i < 14 else "test"
        patient = "repeat" if i < 6 else str(i)
        rows.append({"pair_id": f"p{i}", "target": "hemoglobin_low", "hospital_id": patient,
                     "first_video_id": "early", "second_video_id": "late", "first_value": 10.,
                     "second_value": 11. + i, "raw_delta": 1. + i, "scaled_delta": (1. + i - 6.5) / 5.5,
                     "split": split})
    scaler = {"target": "hemoglobin_low", "unit": "g/L", "fit_split": "train", "median": 6.5, "iqr": 5.5}
    return index, pd.DataFrame(rows), scaler


class DinoPairTests(unittest.TestCase):
    def test_feature_difference_views_weights_and_sampler(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index, records, _ = synthetic_data(root)
            records = records.loc[records.split.eq("train")]
            data = FeaturePairs(index, records, ("original", "hflip", "center_crop", "brightness", "contrast"), root)
            reference = PairedFrameDataset(index, records, views=data.views, index_views=True)
            np.testing.assert_array_equal(data.pair_weights, reference.pair_weights)
            self.assertAlmostEqual(data.pair_weights[6] / data.pair_weights[0], 6.)
            for sample in (0, 3, 5, 17, 99, 100, 1199):
                features, label, frame, weight = data[sample]
                row, view = divmod(sample, 5)
                np.testing.assert_array_equal(features.numpy(), np.full(384, row % 20 + view, np.float32))
                self.assertEqual(frame.item(), row)
                self.assertEqual(label.item(), data.labels[row // 20])
                self.assertEqual(weight.item(), data.pair_weights[row // 20])
            batches = list(DistinctLabViewBatchSampler(data, 240))
            self.assertEqual(len(batches), 5)
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(1200))
            for batch in batches:
                groups = np.asarray(batch).reshape(12, 20)
                self.assertEqual(len(set(groups[:, 0] // 100)), 12)
                self.assertTrue(all(len(set(group % 5)) == 1 for group in groups))

    def test_weighted_microbatch_gradients_match_logical_batch(self):
        torch.manual_seed(7)
        model = nn.Linear(384, 1)
        x, y, weights = torch.randn(240, 384), torch.randn(240), torch.rand(240) + .1
        full = _weighted_loss(model(x).squeeze(1), y, weights)
        full.backward()
        expected = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for start in (0, 120):
            stop = start + 120
            loss = _weighted_loss(model(x[start:stop]).squeeze(1), y[start:stop], weights[start:stop])
            (loss * weights[start:stop].sum() / weights.sum()).backward()
        for parameter, grad in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter.grad, grad, rtol=1e-5, atol=1e-7)

    def test_online_encoder_stays_frozen_for_both_heads(self):
        for width in (32, 64):
            model = FrozenDeltaPredictor(nn.Linear(384, 384), width).train()
            self.assertFalse(model.encoder.training)
            self.assertEqual(sum(p.numel() for p in model.head.parameters()), head_parameters(width))
            first, second = torch.randn(2, 384), torch.randn(2, 384)
            model(first, second).sum().backward()
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()))

    def test_both_heads_train_save_and_pool_original_frames(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            index, records, scaler = synthetic_data(root)
            for width in (32, 64):
                output = root / f"head{width}"
                train_task("hemoglobin_low", width, records, scaler, index, torch.device("cpu"), output,
                           {"backbone_weight_sha256": "test"}, epochs=1, cache_dir=root)
                saved = torch.load(output / "model.pt", weights_only=True)
                self.assertEqual(saved["head_parameters"], head_parameters(width))
                history = pd.read_csv(output / "history.csv")
                self.assertEqual(history.optimizer_steps.iloc[0], 5)
                self.assertEqual(history.train_model_inputs.iloc[0], 1200)
                predictions = pd.read_csv(output / "pair_predictions.csv")
                self.assertTrue(predictions.frame_count.eq(20).all())
                np.testing.assert_array_equal(predictions.y_true, records.raw_delta)


if __name__ == "__main__":
    unittest.main()
