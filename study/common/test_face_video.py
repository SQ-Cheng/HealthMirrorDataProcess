"""Focused compatibility and lossless random-access checks for face224 IO."""

from collections import OrderedDict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pandas as pd
import torch

from preprocess.face_crop_comparison.pipeline import VideoWriter
from study.common.face_video import decode_indexed_frame, resolve_video
from study.exp2_face_pretrained_head32_regression import frame_index as indexing


class FaceVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.env = patch.dict(os.environ, {"HEALTHMIRROR_RAW_ROOT": str(self.root), "HEALTHMIRROR_FACE_SOURCE": "face224"})
        self.env.start()
        self.row = pd.DataFrame([{"video_id": "mirror1_patient_000001", "mirror": "mirror1", "lab_patient_id": 1}])
        self.directory = self.root / "mirror1_data/patient_000001"
        self.directory.mkdir(parents=True)
        (self.directory / "raw_video.avi").write_bytes(b"raw source fixture")
        self.video = self.directory / "face224.mkv"
        self.pixels = np.random.default_rng(42).integers(0, 256, (40, 224, 224, 3), dtype=np.uint8)
        writer = VideoWriter(self.video, 224, 224, lossless=True)
        writer.stream.codec_context.gop_size = 1
        for n, pixels in enumerate(self.pixels):
            writer.write(pixels, .1 + n * .067)
        writer.close()
        mapping = []
        with av.open(str(self.video)) as container:
            for n, frame in enumerate(container.decode(video=0)):
                mapping.append({"output_frame_index": n, "source_frame_index": n * 2 + 5,
                                "source_elapsed_seconds": .1 + n * .067,
                                "encoded_pts": frame.pts, "encoded_time_base": str(frame.time_base)})
        pd.DataFrame(mapping).to_csv(self.directory / "face224_frames.csv", index=False)
        self.metadata = {"status": "completed", "retained_frames": 40, "bytes": self.video.stat().st_size,
                         "video_id": "mirror1_patient_000001", "verification": {"pixel_exact": True},
                         "protocol": {"quality": {"confidence_threshold": .75},
                                      "geometry": {"enable_alignment": False}, "size": [224, 224]},
                         "source_signatures": {}}
        self.save_metadata()

    def save_metadata(self):
        (self.directory / "face224_metadata.json").write_text(json.dumps(self.metadata))

    def tearDown(self):
        self.env.stop()
        self.temporary.cleanup()

    def test_random_packet_decoding_matches_pixels_and_original_indices(self):
        index = indexing.build_or_reuse_frame_index(self.row, self.root / "index")
        self.assertEqual(len(index.starts), 20)
        self.assertGreaterEqual(np.diff(index.source_indices).min(), 2)
        decoders = OrderedDict()
        with self.video.open("rb") as handle:
            for k in np.random.default_rng(7).permutation(20):
                image = decode_indexed_frame(index, 0, int(k), handle, decoders)
                original_output = (int(index.source_indices[k]) - 5) // 2
                np.testing.assert_array_equal(image.permute(1, 2, 0).numpy(), self.pixels[original_output][:, :, ::-1])
        reused = indexing.build_or_reuse_frame_index(self.row, self.root / "index")
        np.testing.assert_array_equal(reused.starts, index.starts)

    def test_allframes_indexes_only_retained_frames(self):
        index = indexing.build_or_reuse_frame_index(self.row, self.root / "all", "allframes")
        self.assertEqual(len(index.starts), 40)
        np.testing.assert_array_equal(index.source_indices, np.arange(40) * 2 + 5)

    def test_no_valid_frames_does_not_fall_back_to_legacy(self):
        self.metadata.update(status="no_valid_frames", retained_frames=0)
        self.video.unlink()
        self.save_metadata()
        pd.DataFrame(columns=["output_frame_index", "source_frame_index"]).to_csv(self.directory / "face224_frames.csv", index=False)
        index = indexing.build_or_reuse_frame_index(self.row, self.root / "empty")
        self.assertEqual(len(index.video_ids), 0)
        self.assertTrue(indexing._index_is_reusable(self.root / "empty", self.row.video_id, "20frame"))

    def test_unready_source_is_not_silently_upsampled(self):
        (self.directory / "face224_metadata.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            resolve_video(next(self.row.itertuples()), self.root)

    def test_missing_raw_source_and_timestamp_failures_are_audited(self):
        (self.directory / "face224_metadata.json").unlink()
        self.video.unlink()
        ledger = self.root / "_face224_processing"
        ledger.mkdir()
        pd.DataFrame({"video_id": ["mirror1_patient_000001"], "status": ["failed"],
                      "error": ["JPEG/timestamp count mismatch: fixture"]}).to_csv(ledger / "index.csv", index=False)
        index = indexing.build_or_reuse_frame_index(self.row, self.root / "bad_times")
        self.assertEqual(len(index.video_ids), 0)
        failures = pd.read_csv(self.root / "bad_times/invalid_frames.csv")
        self.assertIn("timestamp", failures.reason.iloc[0])
        (self.directory / "raw_video.avi").unlink()
        (ledger / "index.csv").unlink()
        index = indexing.build_or_reuse_frame_index(self.row, self.root / "missing_raw")
        self.assertEqual(len(index.video_ids), 0)
        self.assertIn("missing_or_empty", pd.read_csv(self.root / "missing_raw/invalid_frames.csv").reason.iloc[0])

    def test_legacy_indexes_still_load_and_decode(self):
        from io import BytesIO
        from PIL import Image
        from torchvision.io import decode_jpeg, ImageReadMode
        stream = BytesIO()
        Image.fromarray(self.pixels[0, :128, :128]).save(stream, format="JPEG")
        data = stream.getvalue()
        video = self.directory / "video.avi"
        video.write_bytes(data)
        path = self.root / "legacy.npz"
        np.savez(path, video_ids=np.array(["legacy"]), video_paths=np.array([str(video)]),
                 video_ptr=np.array([0, 1]), starts=np.array([0]), ends=np.array([len(data)]), source_indices=np.array([0]))
        index = indexing.FrameOffsetIndex.load(path)
        with video.open("rb") as handle:
            actual = decode_indexed_frame(index, 0, 0, handle, OrderedDict())
        expected = decode_jpeg(torch.frombuffer(bytearray(data), dtype=torch.uint8), mode=ImageReadMode.RGB)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with patch.dict(os.environ, {"HEALTHMIRROR_FACE_SOURCE": "legacy128"}):
            self.assertEqual(resolve_video(next(self.row.itertuples()), self.root), str(video))


if __name__ == "__main__":
    unittest.main()
