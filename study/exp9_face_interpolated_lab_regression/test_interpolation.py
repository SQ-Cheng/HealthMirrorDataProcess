"""CPU-only checks of surgical separation, strict coverage, and no extrapolation."""

import unittest

import numpy as np
import pandas as pd

from .build_dataset import interpolate_video,assign_split,build_records
from . import config
from .preoperative import nearest_preoperative,patient_assignments


class InterpolationTests(unittest.TestCase):
    def setUp(self):
        self.times=np.array([10,20,30,37,50,60,70],float)
        self.values=np.array([1,2,3,1000000,50,60,70],float)

    def label(self,start,end):
        return interpolate_video(self.times,self.values,start,end,35,45,0,100)

    def test_separate_curves_exclude_intraoperative_values(self):
        pre,status=self.label(24,26)
        self.assertEqual(status,"retained")
        self.assertEqual(pre["phase"],"pre")
        self.assertEqual(pre["raw_value"],2.5)
        self.assertEqual(pre["support_right_time_unix"],30)
        post,status=self.label(54,56)
        self.assertEqual(post["phase"],"post")
        self.assertEqual(post["raw_value"],55)
        self.assertEqual(post["support_left_time_unix"],50)

    def test_no_curve_bridge_or_extrapolation(self):
        for start,end,reason in ((34,36,"video_overlaps_cabg_interval"),(36,38,"video_overlaps_cabg_interval"),
                                 (31,33,"pre_no_lab_after_video"),(46,48,"post_no_lab_before_video")):
            value,status=self.label(start,end)
            self.assertIsNone(value)
            self.assertEqual(status,reason)

    def test_strict_before_after_full_video_not_only_midpoint(self):
        value,status=self.label(10,30)
        self.assertIsNone(value)
        self.assertEqual(status,"pre_no_lab_before_video")

    def test_exact_measurement_node_is_preserved_when_both_sides_exist(self):
        value,status=interpolate_video(np.array([10,20,30]),np.array([0,100,10]),18,22,40,50,0,100)
        self.assertEqual(status,"retained")
        self.assertEqual(value["raw_value"],100)
        self.assertEqual(value["at_actual_lab_timestamp"],1)
        self.assertEqual(value["interpolation_alpha"],1)

    def test_24h_bound_is_inclusive_and_applies_to_both_sides(self):
        start=86400.;end=start+60
        args=(start,end,500000.,600000.,-10000.,1000000.)
        value,status=interpolate_video(np.array([0.,86400.,172800.]),np.array([1.,2.,3.]),*args)
        self.assertEqual(status,"retained")
        self.assertEqual(value["coverage_before_delta_h"],24)
        value,status=interpolate_video(np.array([-1.,86400.,172800.]),np.array([1.,2.,3.]),*args)
        self.assertIsNone(value)
        self.assertEqual(status,"pre_before_lab_outside_24h")

    def test_duplicate_timestamps_must_be_collapsed_upstream(self):
        with self.assertRaises(ValueError):
            interpolate_video(np.array([10,10,30]),np.array([1,2,3]),20,22,40,50,0,100)

    def test_reuse_split_does_not_reshuffle_patients(self):
        records=pd.DataFrame({"hospital_id":["a","b","c","d","e","f"],"video_id":range(6),
                              "exp2_split":["train","train","val","val","test","test"],
                              "raw_value":[1,1.1,1.2,1.3,1.4,1.5],"abnormal_score":np.arange(6)/10,
                              "binary_label":[0]*6})
        result,_,_,_=assign_split(records,"lactate_high","reuse_exp2")
        self.assertEqual(result.split.tolist(),records.exp2_split.tolist())

    def test_multiple_cabg_events_are_not_merged_into_one_postoperative_curve(self):
        videos=pd.DataFrame([{"hospital_id":"p","video_id":"v","surgery_start_unix":35.,"surgery_end_unix":45.,"valid_surgery_count":2}])
        references={target:pd.DataFrame({"video_id":["v"]}) for target in config.TARGETS}
        labs=pd.DataFrame(columns=["hospital_id","analyte","timestamp_unix","value"])
        records,audit=build_records(videos,labs,references)
        self.assertTrue(all(frame.empty for frame in records.values()))
        self.assertTrue(audit.status.eq("multiple_cabg_events_in_admission").all())

    def test_preoperative_observed_value_has_no_time_limit_or_bracketing(self):
        hours=3600.
        value,status=nearest_preoperative(np.array([10*hours,201*hours]),np.array([7.,999.]),100*hours,100*hours+60,200*hours,0,300*hours)
        self.assertEqual(status,"retained")
        self.assertEqual(value["raw_value"],7)
        self.assertGreater(value["selected_lab_delta_h"],24)
        self.assertTrue(np.isnan(value["interpolation_alpha"]))

    def test_nearest_preoperative_can_follow_video_but_never_surgery(self):
        value,status=nearest_preoperative(np.array([5.,25.,35.,50.]),np.array([1.,2.,999.,9999.]),20,22,35,0,100)
        self.assertEqual(value["selected_lab_time_unix"],25)
        value,status=nearest_preoperative(np.array([-1.,35.,50.]),np.array([1.,2.,3.]),20,22,35,0,100)
        self.assertIsNone(value);self.assertEqual(status,"pre_no_valid_preoperative_lab")

    def test_new_patients_do_not_reassign_existing_patients(self):
        reference=pd.DataFrame({"hospital_id":["known","known"],"split":["test","test"]})
        existing,extension=patient_assignments("hemoglobin_low",reference,["known","new1","new2","new3"])
        self.assertEqual(existing,{"known":"test"})
        self.assertNotIn("known",extension)
        self.assertEqual(set(extension.values()),{"train","val","test"})
        self.assertEqual((existing,extension),patient_assignments("hemoglobin_low",reference,["new3","new2","new1","known"]))


if __name__=="__main__":unittest.main()
