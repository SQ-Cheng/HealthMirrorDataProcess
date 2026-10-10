"""Check the frame-loss variant without decoding videos or allocating a GPU."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from study.exp2_face_architecture_ablation.train import train_task
from study.exp2_face_architecture_ablation.plots import panels
from study.common.video_loss import VideoSmoothL1Loss
from .run_main_regression import training_config, TARGETS
from . import run_main_regression as main_run


class SyntheticFeatures(Dataset):
    def __init__(self, records, train):
        self.video_records = records.reset_index(drop=True)
        self.frame_video_rows = np.repeat(np.arange(len(records)), 20)
        self.views = 5 if train else 1

    def __len__(self):
        return len(self.frame_video_rows) * self.views

    def __getitem__(self, sample):
        frame, view = divmod(sample, self.views)
        inputs = torch.zeros(384)
        inputs[0] = -1. if frame % 20 < 10 else 1.
        return inputs, torch.tensor(0.), torch.tensor(frame), torch.tensor(view)


def synthetic_loader(index, records, architecture, family, train):
    dataset = SyntheticFeatures(records, train)
    return dataset, DataLoader(dataset, batch_size=len(dataset))


def alternating_model(architecture):
    model = nn.Linear(384, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 0] = 1.
        model.bias.zero_()
    return model


class MainRegressionTests(unittest.TestCase):
    def test_head64_uses_separate_results_and_same_features(self):
        original_output, original_cache = main_run.OUTPUT, main_run.CACHE
        try:
            main_run.configure_variant(64)
            self.assertNotEqual(main_run.OUTPUT, original_output)
            self.assertEqual(main_run.training_config().OUTPUT_DIR, main_run.OUTPUT)
            self.assertEqual(main_run.CACHE, original_cache)
            self.assertEqual(sum(p.numel() for p in main_run.build_main_head(None).parameters()), 24833)
        finally:
            main_run.configure_variant(32)

    def test_frame_losses_are_not_pooled_view_losses(self):
        torch.set_num_threads(1)
        rows = [{"hospital_id": str(i), "video_id": str(i), "split": split,
                 "raw_value": 2., "robust_scaled_raw_value": 0.,
                 "binary_label": i % 2, "score_threshold": 2.}
                for i, split in enumerate(("train", "train", "val", "val", "test", "test"))]
        scaler = {"target": "lactate_high", "unit": "mmol/L", "median": 2., "q1": 1.5,
                  "q3": 2.5, "iqr": 1., "train_videos": 2, "train_video_ids_sha256": "test"}
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "source_records").mkdir()
            pd.DataFrame(rows).to_csv(output / "source_records/lactate_high.csv", index=False)
            cfg = training_config(output, 1)
            cfg.LEARNING_RATES = {"test": 0.}
            cfg.MIN_LEARNING_RATES = {"test": 0.}
            train_task({"architecture": "test", "family": "regression", "target": "lactate_high", "scaler": scaler},
                       None, torch.device("cpu"), experiment_config=cfg, model_factory=alternating_model,
                       loader_factory=synthetic_loader, feature_scaler=lambda *_: None)
            run = output / "regression/test/runs/lactate_high"
            history = pd.read_csv(run / "history.csv")
            self.assertAlmostEqual(history.train_loss.iloc[0], .75)
            self.assertAlmostEqual(history.val_loss.iloc[0], .75)
            self.assertEqual(history.train_model_inputs.iloc[0], 200)
            saved = torch.load(run / "model.pt", weights_only=True)
            self.assertEqual(saved["loss_unit"], "frame")
            prediction = pd.read_csv(run / "video_predictions.csv")
            np.testing.assert_array_equal(prediction.y_pred, np.full(6, 2.))
            self.assertTrue(prediction.frame_count.eq(20).all())
        pooled = VideoSmoothL1Loss()(torch.tensor([-1.] * 10 + [1.] * 10), torch.zeros(20))
        self.assertEqual(pooled.item(), 0.)

    def test_ten_task_figure_layout(self):
        import matplotlib.pyplot as plt
        self.assertEqual(len(TARGETS), 10)
        figure, axes = panels(TARGETS)
        self.assertEqual(axes.shape, (3, 4))
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
