"""Focused checks of modality ordering, gating, packet reuse, and frozen branches."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.common.video_loss import DistinctLabViewBatchSampler
from . import config
from .data import PairedFeatures
from .features import reuse_rows, validate_ir, create_encoders
from .models import build_model
from .run import training_config


class ExperimentTests(unittest.TestCase):
    def test_heads_gradients_and_parameters(self):
        expected = (49281, 109121, 100994, 82049, 174657, 166530)
        for architecture, parameters in zip(config.VARIANTS, expected):
            with self.subTest(architecture=architecture):
                model = build_model(architecture)
                width = sum(config.DIMENSIONS[branch] for branch in config.branches_for(architecture))
                inputs = torch.randn(6, width, requires_grad=True)
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (6, 1))
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
                self.assertEqual(model.head[0].out_features, 32)
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(inputs.grad).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(float(inputs.grad[:, :width//2].abs().sum()), 0)
                self.assertGreater(float(inputs.grad[:, width//2:].abs().sum()), 0)

    def test_gate_initial_balance_and_scalar_broadcast(self):
        for group in config.GROUPS:
            for method, gate_width in (("vector_gate", 64), ("scalar_gate", 1)):
                model = build_model(f"{group}_{method}").eval()
                inputs = torch.randn(4, model.modality_features*2)
                rgb, ir = inputs.split(model.modality_features, dim=1)
                rgb, ir = model.rgb_project(rgb), model.ir_project(ir)
                weight = model.gate(torch.cat((rgb, ir), dim=1)).sigmoid()
                self.assertEqual(tuple(weight.shape), (4, gate_width))
                torch.testing.assert_close(weight, torch.full_like(weight, .5))
                torch.testing.assert_close(model.fuse(inputs), (rgb+ir)/2)
                if gate_width == 1:
                    with torch.no_grad():
                        model.gate.weight.fill_(.02)
                    weight = model.gate(torch.cat((rgb, ir), dim=1)).sigmoid()
                    torch.testing.assert_close(model.fuse(inputs), weight*rgb+(1-weight)*ir)

    def test_four_independent_frozen_modules(self):
        def make(**kwargs):
            return nn.Linear(3, 2).requires_grad_(False).eval()
        with patch("study.exp2_face_rgb_ir_dinov3b_regression.features.load_encoder", side_effect=make), patch(
                "study.exp2_face_rgb_ir_dinov3b_regression.features.load_farl", side_effect=lambda: (make(), [])):
            modules = create_encoders(torch.device("cpu"))
        self.assertEqual(len({p.data_ptr() for model in modules.values() for p in model.parameters()}), 8)
        self.assertTrue(all(not model.training and not any(p.requires_grad for p in model.parameters()) for model in modules.values()))

    def test_ir_validation_no_pseudocolor(self):
        pixels = torch.randint(0, 256, (2, 1, 224, 224), dtype=torch.uint8).repeat(1, 3, 1, 1)
        torch.testing.assert_close(validate_ir(pixels), pixels)
        pixels[0, 1, 0, 0] ^= 1
        with self.assertRaises(ValueError):
            validate_ir(pixels)

    def test_reuse_requires_same_packet_and_source(self):
        def index(starts, sources):
            return SimpleNamespace(video_ids=["a"], video_lookup={"a": 0}, video_paths=["a.mkv"], codec_extradata=["ff"],
                starts=np.array(starts), ends=np.array(starts)+2, source_indices=np.array(sources), frame_range=lambda _: (0, len(starts)))
        original = index([10, 20, 30], [1, 3, 5])
        paired = index([20, 41, 10], [3, 7, 8])
        np.testing.assert_array_equal(reuse_rows(paired, original), [1, -1, -1])
        paired.video_paths = ["other.mkv"]
        np.testing.assert_array_equal(reuse_rows(paired, original), [-1, -1, -1])

    def test_dataset_branch_order_and_batch_coverage(self):
        index = SimpleNamespace(starts=np.arange(240), frame_range=lambda video: (int(video)*20, (int(video)+1)*20))
        records = pd.DataFrame({"video_id": [str(i) for i in range(12)], "clinical_event_id": [f"event{i}" for i in range(12)],
                                "robust_scaled_raw_value": np.arange(12, dtype=np.float32)})
        with tempfile.TemporaryDirectory() as name, patch.object(config, "CACHE", Path(name)):
            for i, branch in enumerate(config.BRANCHES):
                values = np.empty((240, 5, config.DIMENSIONS[branch]), dtype=np.float32)
                values[:] = (np.arange(240)[:, None, None]*10+np.arange(5)[None, :, None]+i*10000)
                np.save(Path(name) / f"{branch}.npy", values)
            dataset = PairedFeatures(index, records, config.VIEWS, "dino_farl_concat")
            features, label, frame, view = dataset[7]
            self.assertEqual((float(label), int(frame), int(view)), (0., 1, 2))
            self.assertEqual([float(features[i]) for i in (0, 768, 1280, 2048)], [12., 20012., 10012., 30012.])
            torch.manual_seed(22)
            batches = list(DistinctLabViewBatchSampler(dataset, 240))
            self.assertEqual([len(batch) for batch in batches], [240]*5)
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(1200))
            for batch in batches:
                rows = np.array(batch).reshape(-1, 20)
                self.assertEqual(len(set(rows[:, 0]//100)), 12)
                self.assertTrue(all(len(set(row % 5)) == 1 for row in rows))

    def test_training_protocol(self):
        cfg = training_config()
        self.assertEqual((cfg.MAX_EPOCHS, cfg.PATIENCE, cfg.WEIGHT_DECAY, cfg.LOSS_LEVEL), (80, 12, .001, "frame"))
        self.assertEqual(set(cfg.LEARNING_RATES.values()), {2e-4})
        self.assertEqual(set(cfg.MIN_LEARNING_RATES.values()), {1e-6})

    def test_automatic_png_figures(self):
        from .plots import plot_results
        targets = ("hemoglobin_low", "egfr_low")
        with tempfile.TemporaryDirectory() as name, patch.object(config, "OUTPUT", Path(name)), patch.object(config, "TARGETS", targets):
            for architecture in config.VARIANTS:
                root = Path(name) / f"regression/{architecture}"
                metrics, history = [], []
                for target in targets:
                    run = root / f"runs/{target}"
                    run.mkdir(parents=True)
                    pd.DataFrame({"hospital_id": ["01", "02", "03", "04"], "video_id": ["a", "b", "c", "d"],
                        "split": "test", "y_true": [1., 2., 3., 4.], "y_pred": [1.2, 1.9, 3.3, 3.8], "frame_count": 20}).to_csv(run / "video_predictions.csv", index=False)
                    metrics.append({"target": target, "split": "test", "mae": .2, "rmse": .25, "pearson_r": .9, "r2": .8})
                    history.extend({"target": target, "epoch": epoch, "train_loss": .5/epoch, "val_loss": .6/epoch,
                                    "val_pearson_r": .4+epoch*.03} for epoch in range(1, 4))
                pd.DataFrame(metrics).to_csv(root / "metrics_all.csv", index=False)
                pd.DataFrame(history).to_csv(root / "history_all.csv", index=False)
            plot_results()
            self.assertEqual(len(list(Path(name).rglob("*.png"))), 17)
            self.assertFalse(list(Path(name).rglob("*.pdf")))
            self.assertEqual(len(pd.read_csv(Path(name) / "paired_test_audit.csv")), 2)


if __name__ == "__main__":
    unittest.main()
