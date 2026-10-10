"""Numerical tests for the 41-dimensional native ROI color baseline."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from .features import FEATURE_NAMES, FEATURE_COUNT, roi_features, frame_features, aggregate_video


def image_with_pixels(pixels):
    pixels = np.asarray(pixels, np.uint8).reshape(-1, 3)
    image = np.full((224, 224, 3), 255, np.uint8)
    mask = np.zeros((224, 224), bool)
    image.reshape(-1, 3)[:len(pixels)] = pixels
    mask.flat[:len(pixels)] = True
    return image, mask


class ColorTests(unittest.TestCase):
    def values(self, pixels):
        rgb, mask = image_with_pixels(pixels)
        with patch("cv2.resize", side_effect=AssertionError("Features must use native pixels")):
            result = roi_features(rgb, mask)
        self.assertEqual(result.shape, (41,))
        self.assertTrue(np.isfinite(result).all())
        return dict(zip(FEATURE_NAMES, result))

    def test_schema_and_mlp_counts(self):
        self.assertEqual(FEATURE_COUNT, 41)
        self.assertEqual(len(set(FEATURE_NAMES)), 41)
        protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
        self.assertEqual(protocol["features"]["dimensions_per_roi"], 41)
        for dimension, parameters in protocol["mlp"]["parameters_by_input_dimension"].items():
            self.assertEqual(parameters, 64 * int(dimension) + 2177)

    def test_native_stats_and_unmasked_pixels_excluded(self):
        values = self.values([[10, 30, 50], [110, 130, 150], [210, 230, 250]])
        self.assertAlmostEqual(values["rgb_r_mean"], 110 / 255, places=6)
        self.assertAlmostEqual(values["rgb_r_std"], np.std([10, 110, 210]) / 255, places=6)
        self.assertAlmostEqual(values["rgb_r_p10"], 30 / 255, places=6)
        self.assertAlmostEqual(values["rgb_r_p50"], 110 / 255, places=6)
        self.assertAlmostEqual(values["rgb_r_p90"], 190 / 255, places=6)
        self.assertAlmostEqual(values["saturated_pixel_fraction"], 1 / 3, places=6)

    def test_rgb_not_bgr_and_log_b_over_g(self):
        red = self.values([[255, 0, 0]])
        self.assertEqual(red["rgb_r_mean"], 1)
        self.assertEqual(red["rgb_b_mean"], 0)
        self.assertGreater(red["lab_a_mean"], 70)
        self.assertGreater(red["lab_b_mean"], 60)
        values = self.values([[20, 80, 160]])
        self.assertAlmostEqual(values["log_b_over_g_mean"], np.log(161 / 81), places=6)
        self.assertLess(values["log_r_over_g_mean"], 0)
        self.assertAlmostEqual(values["chromaticity_r_mean"], 20 / 260, places=6)

    def test_hue_wrap_is_circular(self):
        values = self.values([[255, 4, 0], [255, 0, 4]])
        self.assertGreater(values["hue_weighted_mean_cos"], .99)
        self.assertAlmostEqual(values["hue_weighted_mean_sin"], 0, places=5)
        self.assertGreater(values["hue_concentration"], .99)

    def test_dark_gray_and_saturation_are_finite_and_retained(self):
        black = self.values([[0, 0, 0]])
        self.assertEqual(black["dark_pixel_fraction"], 1)
        self.assertEqual(black["hue_concentration"], 0)
        self.assertEqual(black["chromaticity_r_mean"], 0)
        white = self.values([[255, 255, 255]])
        self.assertEqual(white["saturated_pixel_fraction"], 1)
        self.assertEqual(white["hue_weighted_mean_cos"], 0)
        self.assertAlmostEqual(white["chromaticity_r_mean"], 1 / 3, places=6)
        mixed = self.values([[0, 0, 0], [255, 255, 255]])
        self.assertEqual(mixed["rgb_r_mean"], .5)
        self.assertEqual(mixed["dark_pixel_fraction"], .5)
        self.assertEqual(mixed["saturated_pixel_fraction"], .5)

    def test_invalid_roi_is_not_imputed(self):
        result, valid = frame_features(np.zeros((224, 224, 3), np.uint8), {})
        self.assertEqual(result.shape, (4, 41))
        self.assertTrue(np.isnan(result).all())
        self.assertFalse(valid.any())

    def test_video_aggregation_uses_identical_common_frames(self):
        features = np.arange(20, dtype=np.float32)[:, None, None] * np.ones((20, 4, 41), np.float32)
        valid = np.ones((20, 4), bool)
        valid[0, 0] = False
        features[0, 0] = np.nan
        result, common = aggregate_video(features, valid)
        self.assertEqual(common.sum(), 19)
        np.testing.assert_array_equal(result, np.full((4, 41), 10., np.float32))
        with self.assertRaises(ValueError):
            aggregate_video(features[:9], valid[:9])
        features[1, 1, 0] = np.nan
        with self.assertRaises(ValueError):
            aggregate_video(features, valid)


if __name__ == "__main__":
    unittest.main()
