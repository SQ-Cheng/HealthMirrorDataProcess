"""Augmented native color statistics and leakage-safe five-view training."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from .features import roi_features, color_pixel_features
from .preview_rois import HERE
from .roi import NAMES
from .train import fit_mlp, fit_ridge, target_scale
from .views import augmented_descriptors
from study.exp2_face_pretrained_head32_regression.train import _prepare_images


class FiveViewTests(unittest.TestCase):
    def test_same_augmentation_pixels_and_cheek_order(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        rois = {}
        for j, name in enumerate(NAMES):
            mask = np.zeros((224, 224), dtype=bool)
            mask[20 + j * 30:40 + j * 30, 30:70] = True
            rois[name] = SimpleNamespace(mask=mask)
        original = np.stack([roi_features(rgb, rois[name].mask) for name in NAMES])
        protocol = json.loads((HERE / "protocol.json").read_text())
        params = protocol["features"]["augmentation_parameters"]
        views, valid = augmented_descriptors(rgb, rois, original, np.ones(4, bool), params, protocol["roi"])
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(views[0], original)
        np.testing.assert_array_equal(views[1], original[[0, 2, 1, 3]])
        images = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None]
        augmented = _prepare_images(images, torch.arange(5)[None], "bicubic", torch.device("cpu"), (0, 0, 0), (1, 1, 1)).permute(0, 2, 3, 1).numpy()
        for view in (3, 4):
            for j, name in enumerate(NAMES):
                expected = color_pixel_features(augmented[view][rois[name].mask])
                np.testing.assert_allclose(views[view, j], expected, atol=2e-5, rtol=2e-5)
        self.assertFalse(np.array_equal(views[3], original))
        self.assertFalse(np.array_equal(views[4], original))

    def test_mlp_scaler_uses_all_train_views_only(self):
        x = np.arange(10 * 5 * 2).reshape(10, 5, 2).astype(float)
        y = np.arange(10.)
        options = {"lr": .0005, "weight_decay": .001, "scheduler_factor": .5, "scheduler_patience": 5,
                   "min_lr": .00001, "batch_videos": 64, "max_epochs": 1, "patience": 20, "gradient_clip": 1.}
        with tempfile.TemporaryDirectory() as name:
            _, transform, median, iqr = fit_mlp(x, y, np.ones((2, 2)) * 1e9, np.array([2., 3.]), torch.device("cpu"), options, 2, Path(name))
            np.testing.assert_allclose(transform.mean_, x.reshape(-1, 2).mean(axis=0))
            self.assertEqual((median, iqr), target_scale(y))
            saved = torch.load(Path(name) / "model.pt", weights_only=True)
            self.assertEqual(saved["training_views"], 5)

    def test_ridge_keeps_patient_groups_and_unique_video_target_scaler(self):
        rng = np.random.default_rng(9)
        videos = np.arange(20)
        x = rng.normal(size=(100, 3))
        y = np.repeat(np.linspace(1, 13, 20), 5)
        _, transform, median, iqr, scores = fit_ridge(x, y, np.repeat(videos // 2, 5), [.1, 1],
            video_ids=np.repeat(videos, 5), views=np.tile(np.arange(5), 20))
        self.assertEqual((median, iqr), target_scale(y[::5]))
        self.assertEqual(len(scores), 2)
        np.testing.assert_allclose(transform.mean_, x.mean(axis=0))


if __name__ == "__main__":
    unittest.main()
