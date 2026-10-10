"""Focused checks for the new logical pair batch and weighted accumulation."""

import contextlib
import copy
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.common.video_loss import DistinctLabViewBatchSampler
from .data import PairedFrameDataset
from . import train
from .build_dataset import _pair_target


class FullDataTests(unittest.TestCase):
    def test_twelve_pairs_one_view_all_frames_and_views_once(self):
        count = 25
        records = pd.DataFrame({"pair_id": [f"p{i}" for i in range(count)],
                                "hospital_id": [str(i // 3) for i in range(count)],
                                "first_video_id": ["first"] * count, "second_video_id": ["second"] * count,
                                "scaled_delta": np.linspace(-1, 1, count)})
        index = SimpleNamespace(frame_range=lambda video: (0, 20) if video == "first" else (20, 40))
        dataset = PairedFrameDataset(index, records, train.VIEWS, index_views=True)
        sampler = DistinctLabViewBatchSampler(dataset, 240)
        batches = list(sampler)
        self.assertEqual(len(dataset), count * 100)
        self.assertEqual(dataset.model_input_count, count * 100)
        self.assertEqual(sorted(np.concatenate(batches).tolist()), list(range(count * 100)))
        for batch in batches:
            groups = np.asarray(batch).reshape(-1, 20)
            self.assertEqual(len(set(groups[:, 0] // 100)), len(groups))
            for group in groups:
                self.assertEqual(len(set(group % 5)), 1)
                self.assertEqual(len(set(group // 100)), 1)
        self.assertTrue(all(len(batch) == 240 for batch in batches[:-1]))
        dataset._decode = lambda index: torch.zeros(3, 2, 2, dtype=torch.uint8)
        _, _, label, frame_row, codes, _ = dataset[104]
        self.assertEqual(codes.tolist(), [4])
        self.assertEqual(frame_row.item(), 20)
        self.assertEqual(label.item(), np.float32(records.scaled_delta.iloc[1]))
        dataset.close()

    def test_microbatch_gradient_matches_full_weighted_objective(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.head = torch.nn.Linear(3, 1)
            def forward(self, first, second):
                return self.head((second - first).mean(dim=(-2, -1)))
        torch.manual_seed(10)
        full = Model(); accumulated = copy.deepcopy(full)
        batch = (torch.rand(17, 3, 2, 2), torch.rand(17, 3, 2, 2), torch.rand(17),
                 torch.arange(17), torch.zeros(17, 1, dtype=torch.uint8), torch.linspace(.2, 2, 17))
        stats = []
        with patch.object(train, "_prepare", lambda images, codes, device: images), \
             patch.object(torch, "autocast", lambda **kwargs: contextlib.nullcontext()), \
             patch.object(torch.cuda, "reset_peak_memory_stats"), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(torch.cuda, "max_memory_allocated", return_value=0):
            for model, chunk in ((full, None), (accumulated, 6)):
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                scaler = torch.amp.GradScaler("cpu", init_scale=1024)
                stats.append(train._train_epoch(model, model, [batch], optimizer, scaler, torch.device("cpu"),
                                                 False, None, microbatch_frame_pairs=chunk))
        for first, second in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(stats[0]["loss"], stats[1]["loss"], places=6)
        self.assertEqual(stats[0]["inputs"], 17)

    def test_invalid_closest_crop_cannot_hide_valid_alternative(self):
        base = pd.DataFrame({
            "hospital_id": ["p"] * 4, "video_id": ["invalid", "valid_alternative", "second", "third"],
            "lactate_value": [1., 1., 2., 3.], "lactate_lab_time_unix": [0., 0., 100., 200.],
            "lactate_delta_h": [0., 1., 0., 0.], "capture_time_unix": [0., 1., 100., 200.],
        })
        eligible = base.loc[base.video_id.ne("invalid")]
        records, _ = _pair_target(eligible, "lactate_high", 24)
        self.assertEqual(len(records), 2)
        self.assertEqual(records.iloc[0].first_video_id, "valid_alternative")
        np.testing.assert_array_equal(records.raw_delta, [1., 1.])

    def test_evaluation_preserves_original_raw_delta_precision(self):
        records = pd.DataFrame({
            "pair_id": ["p0", "p1"], "target": ["troponin_high"] * 2, "hospital_id": ["a", "b"],
            "first_video_id": ["a0", "b0"], "second_video_id": ["a1", "b1"],
            "first_value": [0., 20.], "second_value": [16777217., 13.],
            "raw_delta": [16777217., -7.], "lab_interval_h": [1., 1.],
            "first_match_delta_h": [0., 0.], "second_match_delta_h": [0., 0.],
        })
        dataset = SimpleNamespace(records=records, frame_pair_rows=np.repeat(np.arange(2), 20))
        truth = np.repeat(records.raw_delta.to_numpy(np.float32), 20)
        with patch.object(train, "_frame_predictions", return_value=(truth, truth, np.arange(40))):
            _, predictions = train._evaluate(None, dataset, None, None, "test", {"median": 0., "iqr": 1.})
        np.testing.assert_array_equal(predictions.y_true, records.raw_delta)

    def test_all_eleven_tasks_have_result_figures(self):
        from .config import NATIVE_TARGETS
        from .plot_results import plot_results, DISPLAY
        self.assertTrue(set(NATIVE_TARGETS).issubset(DISPLAY))
        with tempfile.TemporaryDirectory(prefix="exp6_eleven_plot_test_") as temporary:
            root = Path(temporary); metrics = []
            for target in NATIVE_TARGETS:
                directory = root / f"runs/{target}"; directory.mkdir(parents=True)
                truth = np.array([-2., 1., 3.]); prediction = np.array([-1., 1., 2.])
                metrics.append({"target": target, "split": "test", **train._metrics(truth, prediction)})
                pd.DataFrame({"target": [target] * 3, "split": ["test"] * 3,
                              "y_true": truth, "y_pred": prediction}).to_csv(directory / "pair_predictions.csv", index=False)
                pd.DataFrame({"target": [target] * 2, "global_epoch": [1, 2], "stage": ["head", "finetune"],
                              "train_mae": [.9, .8], "val_mae": [1., 1.1],
                              "train_pearson_r": [.8, .9], "val_pearson_r": [.6, .65]}).to_csv(directory / "history.csv", index=False)
            pd.DataFrame(metrics).to_csv(root / "metrics_all.csv", index=False)
            plot_results(root)
            self.assertEqual(len(list((root / "figures").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
