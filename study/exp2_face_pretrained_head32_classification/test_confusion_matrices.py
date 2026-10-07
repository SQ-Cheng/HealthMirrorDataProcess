"""Test the video-level count and fixed-threshold contract without a GPU."""

import unittest

import numpy as np
import pandas as pd

from .plot_confusion_matrices import matrix_counts


class ConfusionMatrixTests(unittest.TestCase):
    def test_counts_and_boundary_threshold(self):
        frame = pd.DataFrame({
            "video_id": ["a", "b", "c", "d"], "y_true": [0, 0, 1, 1],
            "y_probability": [0.1, 0.5, 0.2, 0.9], "y_pred": [0, 1, 0, 1],
        })
        np.testing.assert_array_equal(matrix_counts(frame), [[1, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            matrix_counts(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        frame.loc[1, "y_pred"] = 0
        with self.assertRaisesRegex(ValueError, "decision threshold"):
            matrix_counts(frame)

    def test_single_class_keeps_two_by_two_shape(self):
        frame = pd.DataFrame({
            "video_id": ["a", "b"], "y_true": [0, 0],
            "y_probability": [0.1, 0.2], "y_pred": [0, 0],
        })
        np.testing.assert_array_equal(matrix_counts(frame), [[2, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
