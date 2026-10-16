"""Matched20 frame pooling, own-scaler inverse, cohort leakage and PNG checks."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.exp2_face_history_head32_regression.scaling import fit_robust_target_scaler
from . import config
from .compare_rgb import predict_video_values, validate_training_source, plot_comparisons


class ComparisonTests(unittest.TestCase):
    def test_original_twenty_frames_and_checkpoint_scaler(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 2.]]))
        scaler = {"target": "lactate_high", "unit": "mmol/L", "median": 100., "q1": 90., "q3": 110.,
                  "iqr": 20., "train_videos": 2, "train_video_ids_sha256": "test"}
        index = SimpleNamespace(frame_range=lambda video: (20, 40) if video == "later" else (0, 20))
        records = pd.DataFrame({"hospital_id": ["02", "01"], "video_id": ["later", "earlier"],
                                "raw_value": [3., 4.], "score_threshold": 2.})
        first = np.zeros((40, 5, 1), dtype=np.float32)
        first[:, 0, 0] = np.arange(40)
        second = np.ones_like(first)
        # Non-original views must never contaminate test inference.
        first[:, 1:] = 9999
        values = predict_video_values(model, {"state_dict": model.state_dict(), "target_scaler": scaler},
                                      records, index, {"rgb_dino": first, "rgb_farl": second}, ("rgb_dino", "rgb_farl"))
        np.testing.assert_allclose(values.y_pred_scaled, [31.5, 11.5])
        np.testing.assert_allclose(values.y_pred, [730., 330.])
        self.assertTrue(values.frame_count.eq(20).all())
        self.assertEqual(values.video_id.tolist(), ["later", "earlier"])

    def test_source_split_and_own_training_scaler(self):
        records = pd.DataFrame({"hospital_id": ["01", "02", "03", "04"], "video_id": ["a", "b", "c", "d"],
                                "split": ["train", "train", "val", "test"], "raw_value": [1., 3., 2., 4.],
                                "clinical_event_id": ["e1", "e2", "e3", "e4"]})
        target = "lactate_high"
        state = {"target_scaler": fit_robust_target_scaler(target, records, "mmol/L").to_dict(),
                 "seed": config.reference.reference.job_seed("regression", target)}
        validate_training_source(state, records, records.copy(), target)
        changed = records.copy()
        changed.loc[3, "split"] = "train"
        with self.assertRaises(AssertionError):
            validate_training_source(state, records, changed, target)
        leaked = records.copy()
        leaked.loc[3, "hospital_id"] = "01"
        with self.assertRaises(RuntimeError):
            validate_training_source(state, leaked, leaked, target)

    def test_all_eleven_png_comparisons(self):
        rows = []
        for target in ("hemoglobin_low", "egfr_low"):
            for group in config.GROUPS:
                for method in config.METHODS:
                    for modality in ("rgb", "rgb_ir"):
                        rows.append({"target": target, "family": group, "method": method, "modality": modality,
                                     "mae": 1., "rmse": 1.2, "pearson_r": .3 if modality == "rgb" else .4,
                                     "r2": -.1, "explained_variance": .15})
        with tempfile.TemporaryDirectory() as name, patch.object(config, "OUTPUT", Path(name)), patch.object(
                config, "TARGETS", ("hemoglobin_low", "egfr_low")):
            plot_comparisons(pd.DataFrame(rows))
            self.assertEqual(len(list(Path(name).rglob("*.png"))), 11)
            self.assertFalse(list(Path(name).rglob("*.pdf")))


if __name__ == "__main__":
    unittest.main()
