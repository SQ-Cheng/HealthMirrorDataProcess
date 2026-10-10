"""Native mask geometry checks, independent of face detection or training."""

import unittest
from unittest.mock import patch

import numpy as np

from .roi import BROWS, TEMPLES, polygons, polygon_mask, extract_rois


class ROITests(unittest.TestCase):
    def test_forehead_top_is_video_edge(self):
        points = np.full((478, 2), 100., np.float32)
        points[list(BROWS), 0] = np.linspace(40, 184, len(BROWS))
        points[list(BROWS), 1] = 80
        points[list(TEMPLES), 0] = [20, 204]
        options = {"forehead_width_fraction_of_temple_span": .7, "forehead_lower_margin_above_eyebrows_pixels": 3}
        forehead = polygons(points, options)["forehead"][0]
        np.testing.assert_array_equal(forehead[:2, 1], [0, 0])
        np.testing.assert_array_equal(forehead[2:, 1], np.full(len(forehead) - 2, 77))

    def test_inner_mouth_is_excluded(self):
        outer = np.array([[20, 20], [100, 20], [100, 80], [20, 80]], np.float32)
        inner = np.array([[40, 35], [80, 35], [80, 65], [40, 65]], np.float32)
        mask, outside = polygon_mask(outer, inner, (224, 224))
        self.assertEqual(mask[50, 60], 0)
        self.assertEqual(mask[25, 60], 1)
        self.assertEqual(outside, 0.)

    def test_roi_extraction_does_not_resize(self):
        rgb = np.full((224, 224, 3), 123, np.uint8)
        polygon = np.array([[20, 20], [100, 20], [100, 80], [20, 80]], np.float32)
        options = {"cheek_inset_pixels": 2, "minimum_native_valid_pixels_lips": 64,
                   "minimum_native_valid_pixels_skin": 128, "maximum_outside_polygon_fraction": .1}
        with patch("cv2.resize", side_effect=AssertionError("Resize must not be used")), \
             patch("study.exp1_roi_color_regression.roi.polygons", return_value={"forehead": (polygon, None)}):
            roi = extract_rois(rgb, np.zeros((478, 2)), options)["forehead"]
        self.assertEqual(roi.mask.shape, (224, 224))
        self.assertTrue(roi.accepted)
        self.assertEqual(roi.native_pixels, int(roi.mask.sum()))
        np.testing.assert_array_equal(rgb, np.full((224, 224, 3), 123, np.uint8))


if __name__ == "__main__":
    unittest.main()
