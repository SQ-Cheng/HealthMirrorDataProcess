"""Focused tests for the source-boundary-only quality rule."""

import unittest

import numpy as np

from .face_quality import FaceQualityGate, QualityConfig, clipped_crop_bounds
from .mediapipe_alignment import ComparisonConfig, KalmanComparison


class QualityChecks(unittest.TestCase):
    def test_no_alignment_accepts_valid_box_without_valid_eyes(self):
        frame = np.full((480, 640, 3), 100, np.uint8)
        processor = KalmanComparison(ComparisonConfig(enable_alignment=False),
                                     FaceQualityGate(QualityConfig(confidence_threshold=.75)))
        crops, row = processor.process(frame, np.array([[100, 100, 200, 200, .76]]),
                                      np.full((1, 6, 2), np.nan), 0.)
        self.assertTrue(row["direct_eligible"])
        self.assertFalse(row["aligned_eligible"])
        self.assertEqual(row["alignment_status"], "disabled")
        self.assertIsNone(processor.angle_filter)
        self.assertEqual(crops["direct"].shape, (224, 224, 3))
        _, rejected = processor.process(frame, np.array([[100, 100, 200, 200, .74]]),
                                        np.full((1, 6, 2), np.nan), .03)
        self.assertFalse(rejected["direct_eligible"])
        _, held = processor.process(frame, np.empty((0, 5)), np.empty((0, 6, 2)), .06)
        self.assertFalse(held["direct_eligible"])

    def setUp(self):
        self.gate = FaceQualityGate()
        self.frame = np.full((480, 640, 3), 100, np.uint8)
        self.keys = np.array([[130, 130], [180, 130], [155, 150], [155, 180], [100, 150], [200, 150]])

    def test_rejection_does_not_update_kalman(self):
        processor = KalmanComparison(quality_gate=self.gate)
        _, row = processor.process(self.frame, np.array([[100, 100, 200, 200, .4]]), self.keys[None], 0)
        self.assertIn("low_confidence", row["quality_reason"])
        self.assertIsNone(processor.box_filters)
        self.assertIsNone(processor.last_detection_time)
        self.assertFalse(row["aligned_eligible"])

    def test_forehead_outside_rejected_without_clamping(self):
        row = self.gate.assess(self.frame, np.array([100, 10, 300, 210, .95]), self.keys)
        self.assertEqual(row["quality_reason"], "expanded_box_outside_source_over_10_percent")
        self.assertEqual(row["expanded_y1"], -30)

    def test_exact_boundary_contact_allowed(self):
        row = self.gate.assess(self.frame, np.array([0, 80, 640, 480, .95]), self.keys)
        self.assertEqual(row["expanded_y1"], 0)
        self.assertTrue(row["quality_accepted"])

    def test_each_source_edge(self):
        for box in ([-30, 100, 200, 200, .95], [100, 100, 720, 200, .95], [100, 100, 200, 600, .95]):
            with self.subTest(box=box):
                row = self.gate.assess(self.frame, np.array(box), self.keys)
                self.assertIn("expanded_box_outside_source_over_10_percent", row["quality_reason"])

    def test_exact_ten_percent_allowed_and_above_rejected(self):
        exact = self.gate.assess(self.frame, np.array([-10, 120, 90, 220, .95]), self.keys)
        self.assertAlmostEqual(exact["expanded_box_outside_area_fraction"], .1)
        self.assertTrue(exact["quality_accepted"])
        above = self.gate.assess(self.frame, np.array([-10.01, 120, 90, 220, .95]), self.keys)
        self.assertFalse(above["quality_accepted"])

    def test_small_overflow_clipped_not_negative_indexed(self):
        audit = self.gate.assess(self.frame, np.array([-5, 120, 95, 220, .95]), self.keys)
        self.assertTrue(audit["quality_accepted"])
        bounds = clipped_crop_bounds(self.frame, [audit[f"expanded_{key}"] for key in ("x1", "y1", "x2", "y2")])
        self.assertEqual(bounds, (0, 100, 95, 220))

    def test_corner_overflow_uses_intersection_area(self):
        audit = self.gate.assess(self.frame, np.array([-10, 10, 90, 110, .95]), self.keys)
        self.assertAlmostEqual(audit["expanded_box_outside_area_fraction"], .175)
        self.assertFalse(audit["quality_accepted"])

    def test_landmarks_outside_bbox_do_not_reject(self):
        row = self.gate.assess(self.frame, np.array([100, 100, 200, 200, .95]), self.keys + 300)
        self.assertTrue(row["quality_accepted"])

    def test_three_column_preview_keeps_crop_pixels(self):
        from .run_quality_audit import comparison_preview

        aligned = np.full((224, 224, 3), 120, np.uint8)
        direct = np.full((224, 224, 3), 150, np.uint8)
        row = {"aligned_eligible": True, "unfiltered_direct_eligible": True,
               "source_elapsed_seconds": .1, "confidence": .9, "rejection_reason": "accepted",
               "expanded_x1": 100, "expanded_y1": 60, "expanded_x2": 300, "expanded_y2": 300}
        preview = comparison_preview(self.frame, aligned, direct, row, 3)
        self.assertEqual(preview.shape, (416, 896, 3))
        np.testing.assert_array_equal(preview[88:312, 448:672], aligned)
        np.testing.assert_array_equal(preview[88:312, 672:896], direct)
        self.assertTrue((self.frame == 100).all())


if __name__ == "__main__":
    unittest.main()
