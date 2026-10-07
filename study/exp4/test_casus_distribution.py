"""Focused scoring and temporal eligibility checks; no model training."""

import unittest

import numpy as np
import pandas as pd

from study.common.time_alignment import local_naive_to_unix
from .analyze_casus_distribution import CASUS_ANALYTES, _casus_component, _convert_value, match_videos


class CasusDistributionTests(unittest.TestCase):
    def test_units_and_grade_boundaries(self):
        self.assertAlmostEqual(_convert_value("creatinine",88.4,"umol/l"),1)
        self.assertAlmostEqual(_convert_value("bilirubin",17.104,"umol/l"),1)
        self.assertEqual(_convert_value("platelets",120,"10^9/l"),120)
        self.assertTrue(np.isnan(_convert_value("bilirubin",2,"")))
        for name,values in {
            "creatinine":[1.19,1.2,2.3,4.1,5.5,5.50001],
            "bilirubin":[1.19,1.2,3.6,7.1,14,14.00001],
            "lactate":[2.09,2.1,4.1,8.1,12,12.00001],
            "platelets":[121,120,80,50,21,20],
        }.items():
            self.assertEqual([_casus_component(name,value) for value in values],[0,1,2,3,3,4])

    def test_no_preoperative_or_outside_window_labs_and_no_zero_imputation(self):
        start = local_naive_to_unix("2026-01-02 12:00:00")
        video = pd.DataFrame([{
            "video_id":"v","hospital_id":"p","split":"test","recovery_score":.5,
            "index_surgery_end":"2026-01-02 10:00:00","discharge_time":"2026-01-04 10:00:00",
            "capture_start_unix":start,"capture_end_unix":start+60,
        }])
        labs = pd.DataFrame([{"hospital_id":"p","analyte":name,"timestamp_unix":start,"value":1 if name!="platelets" else 200} for name in CASUS_ANALYTES])
        records,audit = match_videos(video,labs,12)
        self.assertEqual(len(records),1)
        self.assertEqual(records.casus_score.iloc[0],0)
        labs.loc[labs.analyte.eq("creatinine"),"timestamp_unix"] = start-3*3600
        records,audit = match_videos(video,labs,12)
        self.assertTrue(records.empty)
        self.assertEqual(audit.missing_components.iloc[0],"creatinine")
        labs.loc[labs.analyte.eq("creatinine"),"timestamp_unix"] = start+60+12*3600
        self.assertEqual(len(match_videos(video,labs,12)[0]),1)
        labs.loc[labs.analyte.eq("creatinine"),"timestamp_unix"] += 1
        self.assertTrue(match_videos(video,labs,12)[0].empty)


if __name__=="__main__":unittest.main()
