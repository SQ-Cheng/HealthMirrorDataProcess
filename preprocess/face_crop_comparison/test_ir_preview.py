"""IR preview crop pixels, quality and source-relative context checks."""

import unittest

import cv2
import numpy as np

from .face_quality import FaceQualityGate
from .mediapipe_alignment import KalmanComparison
from .preview_ir import GEOMETRY, QUALITY, make_preview, target_frame


class IRPreviewTests(unittest.TestCase):
    def test_target_frame_uses_own_elapsed_timestamps(self):
        timestamps = 1000.+np.array([0., .3, .7, 1.2, 1.9])
        self.assertEqual(target_frame(timestamps, 1.), 3)
        self.assertEqual(target_frame(timestamps, 10.), 4)
        self.assertEqual(target_frame(timestamps, 0.), 0)

    def test_grayscale_crop_matches_rgb_production_geometry(self):
        gray = np.tile(np.arange(640, dtype=np.uint16) % 256, (480, 1)).astype(np.uint8)
        frame = np.repeat(gray[..., None], 3, axis=2)
        processor = KalmanComparison(GEOMETRY, FaceQualityGate(QUALITY))
        crop, row = processor.process(frame, np.array([[100, 100, 300, 300, .8]]), np.full((1, 6, 2), np.nan), 0.)
        self.assertTrue(row["direct_eligible"])
        self.assertEqual(row["alignment_status"], "disabled")
        expected = cv2.resize(frame[60:300, 100:300], (224, 224), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(crop["direct"], expected)
        np.testing.assert_array_equal(crop["direct"][..., 0], crop["direct"][..., 2])
        _, rejected = processor.process(frame, np.array([[100, 100, 300, 300, .74]]), np.full((1, 6, 2), np.nan), .03)
        self.assertFalse(rejected["direct_eligible"])

    def test_preview_keeps_original_and_224_crop_pixels(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        original = frame.copy()
        crop = np.full((224, 224, 3), 140, dtype=np.uint8)
        row = {"direct_eligible": True, "source_frame_index": 30, "source_elapsed_seconds": 1.,
               "confidence": .9, "rejection_reason": "accepted"}
        preview = make_preview(frame, crop, row, 1, "mirror1_patient_000001")
        self.assertEqual(preview.shape, (416, 704, 3))
        np.testing.assert_array_equal(preview[88:312, 464:688], crop)
        np.testing.assert_array_equal(frame, original)
        row.update(direct_eligible=False, rejection_reason="low_confidence")
        rejected = make_preview(frame, crop, row, 1, "mirror1_patient_000001")
        self.assertFalse(rejected[88:312, 464:688].any())


if __name__ == "__main__":
    unittest.main()
