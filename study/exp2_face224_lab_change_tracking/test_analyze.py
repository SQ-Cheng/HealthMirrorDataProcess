import unittest

import numpy as np
import pandas as pd

from .analyze import adjacent_pairs, pair_metrics, select_events


class ChangeTrackingTests(unittest.TestCase):
    def test_duplicate_lab_and_reversed_video(self):
        frame = pd.DataFrame({
            "hospital_id": ["1"] * 4, "lab_time_unix": [10, 10, 20, 30],
            "capture_time_unix": [9, 8, 25, 24], "video_id": ["a", "b", "c", "d"],
            "match_delta_h": [0, 1, 0, 0], "midpoint_distance_h": [1, 2, 5, 6],
            "y_true": [1, 1, 2, 3], "y_pred": [2, 999, 3, 4],
        })
        events = select_events(frame)
        self.assertEqual(list(events.video_id), ["a", "c", "d"])
        self.assertEqual(events.matched_videos.iloc[0], 2)
        pairs = adjacent_pairs(events)
        self.assertEqual(list(pairs.included), [True, False])
        self.assertEqual(list(pairs.pred_delta), [1, 1])

    def test_direction_tie_is_not_a_correct_prediction(self):
        frame = pd.DataFrame({"true_delta": [-1., 1., 0.], "pred_delta": [0., 1., 99.]})
        result = pair_metrics(frame)
        self.assertEqual(result["direction_bacc"], .5)
        self.assertEqual(result["direction_accuracy"], .5)

    def test_no_change_is_zero_skill(self):
        result = pair_metrics(pd.DataFrame({"true_delta": [-2., 1., 3.], "pred_delta": [0., 0., 0.]}))
        self.assertEqual(result["mae_skill_vs_no_change"], 0)
        self.assertTrue(np.isnan(result["delta_r"]))


if __name__ == "__main__":
    unittest.main()
