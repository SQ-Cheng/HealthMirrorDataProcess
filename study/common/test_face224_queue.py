"""Check native-only queue readiness and checkpoint reuse without training."""

import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import rerun_face224 as queue


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

    def test_plan_has_no_retired_baseline_dependencies(self):
        self.assertTrue(queue.experiment_plan())
        self.assertTrue(all("baseline" not in job for job in queue.experiment_plan()))
        self.assertTrue(all("face224" in job["output"] for job in queue.experiment_plan()))

    def test_resume_requires_identical_patient_split_and_labels(self):
        import torch
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            records = pd.DataFrame({"pair_id": ["p1", "p2"], "hospital_id": ["1", "2"],
                                    "split": ["train", "test"], "raw_delta": [2., -1.]})
            records.to_csv(root / "records.csv", index=False)
            records.assign(frame_count=20).to_csv(root / "pair_predictions.csv", index=False)
            for name in ("history.csv", "metrics.csv"):
                pd.DataFrame({"value": [1]}).to_csv(root / name, index=False)
            scaler = {"median": 0., "iqr": 1., "target": "hemoglobin_low"}
            torch.save({"target": "hemoglobin_low", "target_scaler": scaler,
                        "model_state_dict": {"weight": torch.ones(1)}}, root / "model.pt")
            job = {"family": "delta", "target": "hemoglobin_low", "scaler": scaler,
                   "run_dir": str(root), "records_path": str(root / "records.csv")}
            self.assertTrue(queue.validate_completed_job(job))
            records.assign(raw_delta=[5., -1.], frame_count=20).to_csv(root / "pair_predictions.csv", index=False)
            with self.assertRaises(AssertionError):
                queue.validate_completed_job(job)

    def test_migrated_face_loader_rejects_legacy_index_before_loading(self):
        from study.exp2_face_pretrained_head32_regression.train import _loader
        index = type("Index", (), {"video_formats": np.array(["mjpeg"])})()
        with self.assertRaisesRegex(ValueError, "128 inputs are retired"):
            _loader(index, pd.DataFrame(), ("original",), "efficientnet_b0", False)


if __name__ == "__main__":
    unittest.main()
