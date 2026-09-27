"""Focused checks for the two-epoch warmup preceding unchanged cosine decay."""

import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from . import train
from .run_patient_diverse_schedule_warmup_ablation import WARMUP_CONFIG


class WarmupScheduleTest(unittest.TestCase):
    def test_stage_adds_warmup_without_shortening_cosine(self):
        model = nn.Linear(1, 1)
        metrics = {
            "mae": 1.0, "rmse": 1.0, "r2": 0.0,
            "pearson_r": 0.0, "spearman_r": 0.0,
            "sign_balanced_accuracy": 0.5, "sign_roc_auc": 0.5,
        }

        def fake_train_epoch(*args, **kwargs):
            args[3].step()
            return 1.0, 1, 1, 1.0, 0.0

        with tempfile.TemporaryDirectory() as run_dir, \
                patch.object(train, "_execution_model", return_value=(model, "eager")), \
                patch.object(train, "_train_epoch", side_effect=fake_train_epoch), \
                patch.object(train, "_evaluate", return_value={"loss": 1.0}), \
                patch.object(train, "_video_metrics", return_value=(metrics, None, None)):
            history = []
            train._run_stage(
                "head", model, model, {"train_augmented": (), "train": (), "val": ()},
                {"train": None, "val": None}, None, torch.device("cpu"),
                1.0, 5, 10, history, run_dir, True, "efficientnet_b0", "test",
                None, None, minimum_learning_rate=0.0, warmup_epochs=2,
            )
        self.assertEqual(len(history), 5)
        self.assertEqual([row["lr_phase"] for row in history],
                         ["warmup", "warmup", "cosine", "cosine", "cosine"])
        for actual, expected in zip(
            [row["learning_rate"] for row in history],
            (0.5, 1.0, 1.0, 0.75, 0.25),
        ):
            self.assertAlmostEqual(actual, expected)

    def test_resolved_ablation_keeps_original_cosine_lengths(self):
        self.assertEqual(WARMUP_CONFIG["head_max_epochs"], 32)
        self.assertEqual(WARMUP_CONFIG["finetune_max_epochs"], 42)
        self.assertEqual(WARMUP_CONFIG["head_warmup_epochs"], 2)
        self.assertEqual(WARMUP_CONFIG["finetune_warmup_epochs"], 2)
        self.assertEqual(WARMUP_CONFIG["head_learning_rate"], 1e-4)
        self.assertEqual(WARMUP_CONFIG["finetune_learning_rate"], 3e-6)
        self.assertEqual(WARMUP_CONFIG["finetune_min_learning_rate"], 1e-7)


if __name__ == "__main__":
    unittest.main()
