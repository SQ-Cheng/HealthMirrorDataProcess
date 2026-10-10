"""Checks of nested matching, assay deduplication, and clinical landmarks."""

import unittest

import numpy as np
import pandas as pd

from .analyze import enrich_timeline, nearest_interval_distance, unique_events
from study.exp2_face_history_head32_regression.source_data import _nearest_measurement


class AttritionTests(unittest.TestCase):
    def test_twelve_hour_inclusive_boundary_and_unchanged_nearest(self):
        start=100000.;end=start+60
        inside=(end+12*3600,1.)
        farther=(end+20*3600,9.)
        self.assertEqual(_nearest_measurement([inside,farther],start,end,12),
                         _nearest_measurement([inside,farther],start,end,24))
        outside=(inside[0]+1,1.)
        self.assertIsNone(_nearest_measurement([outside,farther],start,end,12))
        self.assertIsNotNone(_nearest_measurement([outside,farther],start,end,24))

    def test_event_shared_by_lost_and_retained_pairs_is_not_entirely_lost(self):
        frame=pd.DataFrame([
            {"target":"hemoglobin_low","clinical_event_id":"p@100","hospital_id":"p","raw_value":110.,"binary_label":1,
             "video_id":"a","mirror":"mirror1","retention":"retained_12h"},
            {"target":"hemoglobin_low","clinical_event_id":"p@100","hospital_id":"p","raw_value":110.,"binary_label":1,
             "video_id":"b","mirror":"mirror2","retention":"lost_12_to_24h"},
            {"target":"hemoglobin_low","clinical_event_id":"q@200","hospital_id":"q","raw_value":140.,"binary_label":0,
             "video_id":"c","mirror":"mirror1","retention":"lost_12_to_24h"},
        ])
        result=unique_events(frame).set_index("clinical_event_id")
        self.assertEqual(result.loc["p@100","event_status"],"shared")
        self.assertEqual(result.loc["p@100","mirrors"],"mirror1|mirror2")
        self.assertEqual(result.loc["q@200","event_status"],"lost_only")
        self.assertNotIn("video_id",result.columns)

    def test_landmark_timeline_and_missing_cabg_are_explicit(self):
        pairs=pd.DataFrame([
            {"hospital_id":"p","admission_unix":0.,"discharge_unix":1000.,"label_time_unix":150.,
             "capture_start_unix":10.,"capture_end_unix":20.,"match_signed_delta_h":1.},
            {"hospital_id":"q","admission_unix":0.,"discharge_unix":1000.,"label_time_unix":950.,
             "capture_start_unix":900.,"capture_end_unix":910.,"match_signed_delta_h":1.},
        ])
        episodes=pd.DataFrame([{"hospital_id":"p","admission_unix":0.,"discharge_unix":1000.,
                                "index_surgery_start_unix":100.,"index_surgery_end_unix":200.,"index_surgery_name":"CABG"}])
        result=enrich_timeline(pairs,episodes)
        self.assertEqual(result.lab_nearest_milestone.tolist(),["CABG","CABG unavailable"])
        self.assertEqual(result.lab_cabg_phase.tolist(),["Intra-CABG","CABG unavailable"])
        self.assertEqual(result.video_cabg_phase.tolist(),["Pre-CABG","CABG unavailable"])
        self.assertAlmostEqual(result.lab_stay_fraction.iloc[0],.15)
        self.assertEqual(result.lab_distance_to_cabg_h.iloc[0],0)
        self.assertTrue(np.isnan(result.lab_distance_to_cabg_h.iloc[1]))

    def test_distance_to_surgery_is_zero_inside_interval(self):
        np.testing.assert_array_equal(nearest_interval_distance(np.array([90,100,150,200,210]),100,200),[10,0,0,0,10])


if __name__=="__main__":unittest.main()
