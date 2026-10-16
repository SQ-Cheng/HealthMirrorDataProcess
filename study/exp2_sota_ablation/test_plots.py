"""Incremental PNG regeneration on synthetic predictions, with no optimization."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import config
from .plots import cluster_bootstrap, plot_one_variant, update_comparisons


class PlotTests(unittest.TestCase):
    def test_bootstrap_uses_patient_clusters_and_requires_same_labels(self):
        frame = pd.DataFrame({"hospital_id": ["a", "a", "b", "c", "d"], "video_id": list("12345"),
                              "y_true": [1., 2., 3., 4., 5.], "y_pred": [1.1, 2.1, 3.1, 4.1, 5.1]})
        other = frame.copy()
        other.y_pred += .2
        result = cluster_bootstrap(frame, other, 12, repeats=100)
        self.assertAlmostEqual(result["delta_mae"], .2)
        self.assertAlmostEqual(result["delta_r"], 0.)
        self.assertIn("hospital_id", result["bootstrap_unit"])
        other.y_true += 1
        with self.assertRaises(AssertionError):
            cluster_bootstrap(frame, other, 12, repeats=5)

    def test_incremental_comparisons_create_pngs_and_reuse_unchanged_intervals(self):
        targets = ("hemoglobin_low", "egfr_low")
        name = "a01_regularized_rgb_prior"
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "OUTPUT", Path(directory)/"out"), patch.object(
                config, "BASELINE", Path(directory)/"base"), patch.object(config, "TARGETS", targets), patch(
                "study.exp2_sota_ablation.run.completed_variant", side_effect=lambda variant: variant == name):
            config.BASELINE.mkdir()
            config.OUTPUT.mkdir()
            (config.BASELINE / "target_scalers.json").write_text(json.dumps({target: {"iqr": 2.} for target in targets}))
            roots = (config.BASELINE / f"regression/{config.ARCHITECTURE}", config.OUTPUT / name)
            for root in roots:
                all_history = []
                for target in targets:
                    run = root / "runs" / target
                    run.mkdir(parents=True)
                    prediction = pd.DataFrame({"hospital_id": ["a", "b", "c", "d"], "video_id": list("1234"),
                        "split": "test", "y_true": [1., 2., 3., 4.], "y_pred": [1.1, 2.1, 3.1, 4.1], "frame_count": 20})
                    prediction.to_csv(run / "video_predictions.csv", index=False)
                    pd.DataFrame([{"target": target, "split": split, "mae": .1, "rmse": .1,
                                   "pearson_r": 1., "r2": .99} for split in ("train", "val", "test")]).to_csv(run / "metrics.csv", index=False)
                    all_history.extend({"target": target, "epoch": epoch, "train_loss": .1/epoch,
                                        "val_loss": .2/epoch, "val_pearson_r": .3+epoch*.01} for epoch in range(1, 4))
                pd.DataFrame(all_history).to_csv(root / "history_all.csv", index=False)
            plot_one_variant(name)
            update_comparisons()
            with patch("study.exp2_sota_ablation.plots.cluster_bootstrap") as resample:
                update_comparisons()
                resample.assert_not_called()
            self.assertEqual(len(list(config.OUTPUT.rglob("*.png"))), 9)
            self.assertFalse(list(config.OUTPUT.rglob("*.pdf")))
            winners = pd.read_csv(config.OUTPUT / "tables/best_by_validation_only.csv")
            self.assertEqual(len(winners), 2)
            self.assertEqual(len(pd.read_csv(config.OUTPUT / "tables/test_cohort_audit.csv")), 4)


if __name__ == "__main__":
    unittest.main()
