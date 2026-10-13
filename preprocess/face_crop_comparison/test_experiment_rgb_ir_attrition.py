"""Attrition counts preserve labels/splits and do not bridge lost endpoints."""

import unittest

import pandas as pd

from .audit_experiment_rgb_ir_attrition import filter_records, summary_row


class AttritionTests(unittest.TestCase):
    def test_exp2_patient_and_lab_counts_are_not_video_counts(self):
        records = pd.DataFrame({"hospital_id": ["p1", "p1", "p2"], "video_id": ["a", "b", "c"],
                                "clinical_event_id": ["p1@1", "p1@1", "p2@2"], "split": "train"})
        eligible = {"a"}
        retained = records.loc[filter_records(records, "exp2", eligible)]
        row = summary_row(records, retained, "exp2", "test_target", "train", eligible)
        self.assertEqual(row["lost_samples"], 2)
        self.assertEqual(row["lost_patients"], 1)
        self.assertEqual(row["lost_unique_lab_events"], 1)
        self.assertEqual(row["frame_or_temporal_pair_inputs_rgb_ir"], 100)

    def test_exp6_requires_both_original_endpoints_without_bridging(self):
        records = pd.DataFrame({"hospital_id": ["p1", "p1"], "first_video_id": ["a", "b"],
                                "second_video_id": ["b", "c"], "first_lab_time_unix": [1., 2.],
                                "second_lab_time_unix": [2., 3.], "split": "test"})
        eligible = {"a", "c"}
        retained = records.loc[filter_records(records, "exp6", eligible)]
        self.assertEqual(len(retained), 0)
        row = summary_row(records, retained, "exp6", "test_target", "test", eligible)
        self.assertEqual(row["eligible_videos_orphaned_by_partner_loss"], 2)
        self.assertEqual(row["lost_patients"], 1)
        self.assertEqual(row["lost_unique_lab_events"], 3)


if __name__ == "__main__":
    unittest.main()
