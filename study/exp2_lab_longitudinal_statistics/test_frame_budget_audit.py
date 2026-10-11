"""Spacing capacity, current-policy edge cases and adaptive frame accounting."""

from itertools import combinations
import unittest

import numpy as np
import pandas as pd

from .frame_budget_audit import spaced_positions, uniform_positions, scenario_metrics


class FrameBudgetTests(unittest.TestCase):
    def test_greedy_capacity_is_maximum(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            indices = np.sort(rng.choice(20, 8, replace=False))
            optimal = max(len(group) for size in range(9) for group in combinations(indices, size)
                          if size < 2 or np.min(np.diff(group)) >= 2)
            self.assertEqual(len(spaced_positions(indices)), optimal)

    def test_original_frame_gap_not_retained_position_gap(self):
        indices = np.arange(20) * 3
        self.assertIsNotNone(uniform_positions(indices, 20))
        self.assertIsNone(uniform_positions(np.arange(38), 20))
        self.assertIsNotNone(uniform_positions(np.arange(39), 20))
        self.assertIsNotNone(uniform_positions(np.arange(20), 20, gap=1))

    def test_current_uniform_can_fail_despite_sufficient_capacity(self):
        indices = np.r_[np.arange(5), np.arange(100, 160, 3)]
        self.assertIsNone(uniform_positions(indices, 20))
        spaced = spaced_positions(indices)
        self.assertGreaterEqual(len(spaced), 20)
        self.assertIsNotNone(uniform_positions(indices[spaced], 20))

    def test_fixed_and_adaptive_totals_are_different(self):
        frame = pd.DataFrame({"video_id": ["a", "b"], "hospital_id": ["p", "q"],
                              "valid_face_frames": [100, 7], "max_gap2_frames": [50, 5],
                              "validated_completed_crop": True, "current_n20": [True, False]})
        from .frame_budget_audit import THRESHOLDS
        for count in THRESHOLDS:
            for policy in ("current", "constrained", "gap1"):
                frame[f"{policy}_n{count}"] = frame.max_gap2_frames.ge(count)
        result = scenario_metrics(frame, {"scope": {"a", "b"}})
        fixed = result.loc[result.policy.eq("current") & result.frames_per_video.eq(5)].iloc[0]
        adaptive = result.loc[result.policy.eq("adaptive_cap20") & result.frames_per_video.eq(5)].iloc[0]
        self.assertEqual(fixed.selected_original_frames, 10)
        self.assertEqual(adaptive.selected_original_frames, 25)
        self.assertEqual(adaptive.all_retained_original_frames_in_admitted_videos, 107)
        self.assertEqual(adaptive.added_patients_vs_current20, 1)


if __name__ == "__main__":
    unittest.main()
