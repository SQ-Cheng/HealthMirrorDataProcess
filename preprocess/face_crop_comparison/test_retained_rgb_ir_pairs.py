"""Paired 20-frame selection and modality-safe production paths."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from .count_retained_rgb_ir_pairs import select20, count_video
from .process_full_dataset import output_prefix, summary_directory


class RetainedPairTests(unittest.TestCase):
    def test_rgb_and_ir_outputs_cannot_collide(self):
        self.assertEqual(output_prefix({}), "face224")
        self.assertEqual(output_prefix({"modality": "ir"}), "face224_ir")
        self.assertEqual(summary_directory(Path("/tmp"), "rgb"), Path("/tmp/_face224_processing"))
        self.assertEqual(summary_directory(Path("/tmp"), "ir"), Path("/tmp/_face224_ir_processing"))

    def test_twenty_sparse_pairs_are_valid(self):
        rgb, ir = np.arange(20)*3, np.arange(20)*4
        selected, maximum = select20(rgb, ir)
        self.assertEqual(len(selected), 20)
        self.assertEqual(maximum, 20)
        self.assertTrue((np.diff(rgb[selected]) >= 2).all())
        self.assertTrue((np.diff(ir[selected]) >= 2).all())

    def test_adjacent_pairs_need_sufficient_nonadjacent_capacity(self):
        selected, maximum = select20(np.arange(38), np.arange(38))
        self.assertEqual(len(selected), 0)
        self.assertEqual(maximum, 19)
        selected, maximum = select20(np.arange(39), np.arange(39))
        self.assertEqual(len(selected), 20)
        self.assertEqual(maximum, 20)

    def test_both_modalities_need_gaps_and_ir_cannot_repeat(self):
        selected, maximum = select20(np.arange(20)*3, np.arange(20))
        self.assertEqual(len(selected), 0)
        self.assertEqual(maximum, 10)
        with self.assertRaises(ValueError):
            select20(np.array([0, 2, 4]), np.array([0, 0, 2]))

    def test_video_with_no_valid_ir_frames_is_counted_not_failed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            directory = root / "mirror1_data/patient_000001"
            directory.mkdir(parents=True)
            times = np.array([1000., 1000.03, 1000.06])
            rgb = pd.DataFrame({"source_frame_index": [0, 1, 2], "source_recorder_timestamp_unix": times,
                                "canonical_session_frame_time_unix": times})
            ir = pd.DataFrame(columns=["source_frame_index", "output_frame_index", "source_recorder_timestamp_unix"])
            audit = pd.DataFrame({"source_frame_index": [0, 1, 2], "rejection_reason": ["no_detection"]*3})
            audit.to_csv(directory / "face224_ir_audit.csv", index=False)
            metadata = {"source_frames": 3, "source_signatures": {}}
            job = {"baseline": {"video_id": "mirror1_patient_000001", "mirror": "mirror1",
                    "rgb_path": str(directory / "face224.mkv"), "rgb_retained_frames": 3,
                    "mutual_nearest_frames_33.333ms": 3, "ir_structure_status": "consistent_jpeg_timestamp_index"},
                   "ir_record": {"status": "no_valid_frames"}, "raw_root": str(root), "ir_root": str(root)}
            module = "preprocess.face_crop_comparison.count_retained_rgb_ir_pairs"
            with patch(module+".retained_mapping", side_effect=[(rgb, {}), (ir, metadata)]), patch(module+".RawRecording", return_value=SimpleNamespace(timestamps=times, close=Mock())):
                result, selected, reasons = count_video(job)
            self.assertEqual(result["candidate_pairs"], 3)
            self.assertEqual(result["retained_pairs"], 0)
            self.assertFalse(result["pairs20_eligible"])
            self.assertEqual(selected, [])
            self.assertEqual(reasons, {"no_detection": 3})


if __name__ == "__main__":
    unittest.main()
