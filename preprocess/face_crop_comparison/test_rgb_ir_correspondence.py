"""Temporal correspondence is not ordinal alignment or a many-to-one join."""

import unittest

import numpy as np

from .audit_rgb_ir_correspondence import nearest_indices


class CorrespondenceTests(unittest.TestCase):
    def test_ties_choose_earlier_and_boundaries_are_measured(self):
        indices, distances = nearest_indices(np.array([-1., 1., 2., 3., 5.]), np.array([0., 2., 4.]))
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(distances, [1000., 1000., 0., 1000., 1000.])

    def test_mutual_nearest_does_not_reuse_an_ir_frame(self):
        rgb = np.array([0., .02, .10])
        ir = np.array([.01, .12])
        chosen, _ = nearest_indices(rgb, ir)
        inverse, _ = nearest_indices(ir, rgb)
        mutual = inverse[chosen] == np.arange(len(rgb))
        self.assertEqual(chosen.tolist(), [0, 0, 1])
        self.assertEqual(mutual.tolist(), [True, False, True])
        self.assertEqual(len(np.unique(chosen[mutual])), int(mutual.sum()))

    def test_use_absolute_clocks_not_independent_elapsed_origins(self):
        rgb = np.array([1000., 1000.03])
        ir = np.array([1001., 1001.03])
        _, absolute_distance = nearest_indices(rgb, ir)
        self.assertTrue((absolute_distance > 900).all())
        _, misleading_distance = nearest_indices(rgb-rgb[0], ir-ir[0])
        self.assertTrue((misleading_distance < 1e-9).all())

    def test_invalid_reference_is_rejected(self):
        for reference in (np.array([]), np.array([1., 1.]), np.array([2., 1.]), np.array([np.nan])):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                nearest_indices(np.array([1.]), reference)


if __name__ == "__main__":
    unittest.main()
