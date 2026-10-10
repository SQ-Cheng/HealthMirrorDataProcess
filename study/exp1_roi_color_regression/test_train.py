"""Small training checks without full extraction or formal output artifacts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .train import MLP, fit_ridge, fit_mlp


class TrainingTests(unittest.TestCase):
    def test_mlp_parameters(self):
        for dimension, count in ((41, 4801), (82, 7425), (164, 12673)):
            self.assertEqual(sum(p.numel() for p in MLP(dimension).parameters()), count)

    def test_ridge_and_mlp_save_real_predictions(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=(40, 41))
        y = 3 * x[:, 0] + rng.normal(size=40) * .1
        model, transform, median, iqr, cv = fit_ridge(x[:30], y[:30], np.repeat(np.arange(10), 3), [.01, 1.])
        self.assertEqual(len(cv), 2)
        self.assertTrue(np.isfinite(model.predict(transform.transform(x))).all())
        options = {"lr": .0005, "weight_decay": .001, "scheduler_factor": .5, "scheduler_patience": 5,
                   "min_lr": .00001, "batch_videos": 64, "max_epochs": 2, "patience": 20, "gradient_clip": 1.}
        with tempfile.TemporaryDirectory() as name:
            fitted, *_ = fit_mlp(x[:30], y[:30], x[30:], y[30:], torch.device("cpu"), options, 10, Path(name))
            self.assertTrue((Path(name) / "model.pt").is_file())
            self.assertTrue(np.isfinite(fitted(torch.zeros(1, 41)).detach().numpy()).all())


if __name__ == "__main__":
    unittest.main()
