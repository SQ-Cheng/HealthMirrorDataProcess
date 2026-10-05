"""Check retained-cohort handling and paired comparison output without training."""

import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import rerun_face224 as queue
from .plot_face224_comparison import plot_comparison


class QueueTests(unittest.TestCase):
    def test_readiness_waits_for_lock_and_rejects_unknown_failures(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            directory = root / "_face224_processing"
            directory.mkdir()
            (root / "mirror1_data/patient_000001").mkdir(parents=True)
            (directory / "protocol.json").write_text(json.dumps({
                "geometry": {"enable_alignment": False},
                "quality": {"confidence_threshold": .75},
            }))
            ledger = pd.DataFrame({"video_id": ["mirror1_patient_000001"],
                                   "status": ["failed"],
                                   "error": ["JPEG/timestamp count mismatch: fixture"]})
            ledger.to_csv(directory / "index.csv", index=False)
            with patch.object(queue, "raw_root", return_value=root), patch.object(queue, "STATE", root / "state"):
                with (directory / ".processing.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    self.assertFalse(queue.preprocessing_ready())
                self.assertTrue(queue.preprocessing_ready())
                self.assertTrue((root / "state/preprocessing_exclusions.csv").is_file())
                ledger["error"] = "Unexpected decoder or encoding failure"
                ledger.to_csv(directory / "index.csv", index=False)
                with self.assertRaisesRegex(RuntimeError, "unfinished/failed"):
                    queue.preprocessing_ready()
                ledger["status"] = "pending"
                ledger.to_csv(directory / "index.csv", index=False)
                with self.assertRaisesRegex(RuntimeError, "unfinished/failed"):
                    queue.preprocessing_ready()

    def test_prepare_preserves_split_and_refits_only_retained_train(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            baseline, output = root / "baseline", root / "candidate"
            (baseline / "source_data").mkdir(parents=True)
            (root / "index").mkdir()
            (root / "index/frame_offsets.npz").write_bytes(b"test index")
            pd.DataFrame({"video_id": ["v3"], "status": ["excluded"], "reason": ["too few valid frames"]}).to_csv(root / "index/video_frame_summary.csv", index=False)
            frame = pd.DataFrame({
                "hospital_id": ["1", "2", "3", "4", "5", "6"],
                "video_id": ["v1", "v2", "v3", "v4", "v5", "v6"],
                "split": ["train", "train", "train", "val", "test", "test"],
                "raw_value": [100., 120., 900., 110., 80., 130.],
                "robust_scaled_raw_value": [0.] * 6,
            })
            index = type("Index", (), {"video_lookup": {key: i for i, key in enumerate(["v1", "v2", "v4", "v5", "v6"])}})()
            job = {"key": "fixture", "family": "regression", "output": str(output), "baseline": str(baseline)}
            with patch.object(queue, "INDEX_DIR", root / "index"), patch.object(queue, "targets_for", return_value=("hemoglobin_low",)), patch.object(queue, "load_reference", return_value=frame):
                scalers = queue.prepare_experiment(job, index)
            self.assertEqual(scalers["hemoglobin_low"].median, 110.)
            saved = pd.read_csv(output / "task_records/hemoglobin_low.csv")
            self.assertEqual(set(saved.video_id), {"v1", "v2", "v4", "v5", "v6"})
            self.assertEqual(saved.set_index("video_id").split.to_dict(), frame.set_index("video_id").split.drop("v3").to_dict())
            self.assertEqual(len(pd.read_csv(output / "excluded_records.csv")), 1)

    def test_binary_comparison_uses_same_held_out_identities(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            old, new = root / "old", root / "new"
            for output in (old, new):
                (output / "runs/efficientnet_b0/hemoglobin_low").mkdir(parents=True)
            pd.DataFrame({"target": ["hemoglobin_low"], "split": ["test"]}).to_csv(old / "metrics_all.csv", index=False)
            values = pd.DataFrame({"hospital_id": ["1", "2", "3"], "video_id": ["v1", "v2", "v3"],
                                   "split": ["test"] * 3, "y_true": [0, 1, 1], "y_probability": [.1, .8, .6]})
            values.to_csv(old / "runs/efficientnet_b0/hemoglobin_low/video_predictions.csv", index=False)
            values.iloc[:2].assign(y_probability=[.3, .7]).to_csv(new / "runs/efficientnet_b0/hemoglobin_low/video_predictions.csv", index=False)
            pd.DataFrame({"target": ["hemoglobin_low"], "split": ["test"],
                          "balanced_accuracy": [1.], "roc_auc": [1.], "f1": [1.],
                          "average_precision": [1.]}).to_csv(new / "metrics_all.csv", index=False)
            pd.DataFrame({"target": ["hemoglobin_low"] * 2, "global_epoch": [1, 2], "train_loss": [.4, .2], "val_loss": [.5, .4]}).to_csv(new / "history_all.csv", index=False)
            plot_comparison({"key": "fixture", "family": "classification", "baseline": str(old), "output": str(new)})
            results = pd.read_csv(new / "face224_comparison.csv")
            self.assertEqual(set(results.loc[results.cohort.eq("common_test"), "n"]), {2})
            self.assertEqual(set(results.loc[results.cohort.eq("full_test"), "n"]), {2, 3})
            self.assertTrue((new / "figures/training_history.png").is_file())
            self.assertTrue((new / "figures/test_classification_metrics.png").is_file())


if __name__ == "__main__":
    unittest.main()
