"""Check the frame-loss variant without decoding videos or allocating a GPU."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

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
    def test_referenced_rgb_audit_ignores_only_unused_failed_directories(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            video, sidecar = root / "face224.mkv", root / "frames.csv"
            video.write_bytes(b"rgb-video"); sidecar.write_bytes(b"frame-mapping")
            stat, extra = video.stat(), sidecar.stat()
            manifest = {"schema_version": 4, "face_source": "face224", "frame_policy": {"frames_per_video": 20},
                "videos": [{"video_id": "v", "video_path": str(video), "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "sidecars": {str(sidecar): [extra.st_size, extra.st_mtime_ns]}}],
                "failed_videos": [{"video_id": "unused", "fingerprint_path": str(root), "mtime_ns": 0}]}
            (root / "index_manifest.json").write_text(json.dumps(manifest))
            index = SimpleNamespace(video_ids=np.array(["v"]), video_paths=np.array([str(video)]))
            with patch.object(main_run, "INDEX_PATH", root / "frame_offsets.npz"):
                main_run.validate_referenced_sources(index)
                sidecar.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "sidecar changed"):
                    main_run.validate_referenced_sources(index)

    def test_backbone_size_plots_require_identical_test_labels(self):
        from .plots import plot_backbone_size_comparison
        from . import config as dino_config
        from study.exp2_face_pretrained_head32_regression.train import _regression_metrics
        target = "hemoglobin_low"
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            small, base, baseline = root / "small", root / "base", root / "baseline"
            baseline.mkdir()
            (baseline / "target_scalers.json").write_text(json.dumps({"targets": {target: {"iqr": 30.}}}))
            for output, predictions in ((small, [100., 116., 130.]), (base, [101., 117., 129.])):
                run = output / f"runs/{target}"
                run.mkdir(parents=True)
                values = pd.DataFrame({"split": ["test"]*3, "hospital_id": ["0001", "0002", "0003"],
                    "video_id": ["v1", "v2", "v3"], "y_true": [95., 115., 135.], "y_pred": predictions,
                    "frame_count": [20]*3})
                values.to_csv(run / "video_predictions.csv", index=False)
                metrics = _regression_metrics(values.y_true, values.y_pred, np.full(3, 130.), "low")
                pd.DataFrame([{**metrics, "target": target, "split": "test"}]).to_csv(output / "metrics_all.csv", index=False)
            with patch.object(dino_config, "HERE", root):
                plot_backbone_size_comparison(base, small, baseline, (target,))
                self.assertTrue((root / "outputs/figures/backbone_size_comparison.png").is_file())
                self.assertEqual(len(pd.read_csv(root / "outputs/backbone_size_test_comparison.csv")), 2)
                path = base / f"runs/{target}/video_predictions.csv"
                changed = pd.read_csv(path); changed.loc[0, "y_true"] += 1
                changed.to_csv(path, index=False)
                with self.assertRaises(AssertionError):
                    plot_backbone_size_comparison(base, small, baseline, (target,))

    def test_vitb_head32_is_separate_and_does_not_change_batch_rng(self):
        try:
            main_run.configure_variant(32)
            small_output, small_cache = main_run.OUTPUT, main_run.CACHE
            torch.manual_seed(12)
            small = main_run.build_main_head(None)
            expected_rng = torch.get_rng_state().clone()
            main_run.configure_variant(32, "dinov3_vitb16")
            torch.manual_seed(12)
            base = main_run.build_main_head(None)
            self.assertEqual(base[0].in_features, 768)
            self.assertEqual(base[0].out_features, 32)
            self.assertEqual(sum(p.numel() for p in base.parameters()), 24705)
            self.assertEqual(sum(p.numel() for p in small.parameters()), 12417)
            self.assertNotEqual(main_run.OUTPUT, small_output)
            self.assertNotEqual(main_run.CACHE, small_cache)
            self.assertEqual(main_run.training_config().LEARNING_RATES[main_run.ARCHITECTURE], 2e-4)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        finally:
            main_run.configure_variant(32)

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
