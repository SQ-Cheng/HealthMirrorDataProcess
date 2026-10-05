"""Focused CPU tests for native 12h five-fold scheduling and source control."""

import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import run_face224_12h_5fold as runner
from . import selected_5fold_splits as splits
from .plot_face224_12h_5fold import plot_oof, plot_classification_comparison


class FiveFoldTests(unittest.TestCase):
    def test_resolved_schedules(self):
        expected = {
            "head_learning_rate": 1e-4, "head_min_learning_rate": 1e-6,
            "head_max_epochs": 30, "head_patience": 8,
            "finetune_learning_rate": 3e-6, "finetune_min_learning_rate": 1e-7,
            "finetune_max_epochs": 40, "finetune_patience": 8,
        }
        self.assertEqual(runner.schedule_for("regression_diverse"), expected)
        self.assertEqual(runner.schedule_for("classification_diverse"), expected)
        standard = runner.schedule_for("classification_standard")
        self.assertEqual(standard["head_learning_rate"], 2e-4)
        self.assertEqual(standard["finetune_learning_rate"], 1e-5)
        self.assertEqual((standard["head_max_epochs"], standard["finetune_max_epochs"]), (40, 60))
        self.assertEqual((standard["head_patience"], standard["finetune_patience"]), (10, 12))

    def test_dependency_requires_both_released_lock_and_completion(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with (d / ".lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                (d / "COMPLETE").touch()
                self.assertFalse(runner.dependency_finished(d, ".lock"))
            self.assertTrue(runner.dependency_finished(d, ".lock"))
            (d / "COMPLETE").unlink()
            with self.assertRaisesRegex(RuntimeError, "before completion"):
                runner.dependency_finished(d, ".lock")

    def test_custom_source_reuses_existing_fold_algorithm_without_leakage(self):
        patients = np.repeat(np.arange(20), 2)
        records = pd.DataFrame({
            "hospital_id": patients.astype(str), "video_id": [f"v{i}" for i in range(40)],
            "binary_label": np.repeat(np.arange(20) % 2, 2),
            "raw_value": 90 + patients * 2., "abnormal_score": (patients - 10.) / 3,
        })
        target = "hemoglobin_low"
        with tempfile.TemporaryDirectory() as d, patch.object(splits, "TARGETS", (target,)):
            root = Path(d) / "splits"
            original = splits.choose_folds
            with patch.object(splits, "choose_folds", side_effect=lambda r, t: original(r, t, candidates=3)):
                splits.prepare_splits(lambda _: (records.copy(), ["fixture_sha"]), root, {"hours": 12})
            tests = []
            for fold in range(5):
                saved = pd.read_csv(root / f"{target}_fold{fold}.csv", dtype={"hospital_id": str})
                self.assertEqual(saved.groupby("hospital_id").split.nunique().max(), 1)
                tests.extend(saved.loc[saved.split.eq("test"), "video_id"])
            self.assertEqual(len(tests), len(records))
            self.assertEqual(set(tests), set(records.video_id))
            splits.prepare_splits(lambda _: (records.copy(), ["fixture_sha"]), root, {"hours": 12})
            with self.assertRaisesRegex(RuntimeError, "source policy"):
                splits.prepare_splits(lambda _: (records.copy(), ["fixture_sha"]), root, {"hours": 24})

    def test_oof_and_paired_classification_plots(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            outputs = {p: d / p for p in ("regression_diverse", "classification_diverse", "classification_standard")}
            for protocol, root in outputs.items():
                root.mkdir()
                rows = []
                for target in runner.TARGETS:
                    for fold in range(5):
                        for i in range(2):
                            row = {"target": target, "fold": fold, "hospital_id": str(fold * 2 + i), "video_id": f"v{fold}_{i}", "y_true": i}
                            row["y_pred" if protocol == "regression_diverse" else "y_probability"] = .1 + i * .8
                            rows.append(row)
                pd.DataFrame(rows).to_csv(root / "oof_predictions.csv", index=False)
                for fold in range(5):
                    (root / f"fold_{fold}/figures").mkdir(parents=True)
                    pd.DataFrame({"target": runner.TARGETS, "global_epoch": [1] * 8,
                                  "train_loss": [.2] * 8, "val_loss": [.3] * 8}).to_csv(root / f"fold_{fold}/history_all.csv", index=False)
                plot_oof(root, protocol)
                self.assertTrue((root / "figures" / ("oof_predicted_vs_true.png" if protocol == "regression_diverse" else "oof_roc_curves.png")).is_file())
                if protocol != "regression_diverse":
                    pd.DataFrame([{"target": target, "metric": metric, "fold_mean": .8,
                                   "fold_std": .02, "pooled_oof": .82}
                                  for target in runner.TARGETS for metric in ("roc_auc", "balanced_accuracy")]).to_csv(root / "cv_summary.csv", index=False)
                (root / "COMPLETE").touch()
            plot_classification_comparison(outputs, d / "figures")
            self.assertTrue((d / "figures/classification_roc_auc_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
