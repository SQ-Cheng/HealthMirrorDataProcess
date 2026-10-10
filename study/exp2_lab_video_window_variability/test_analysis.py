"""Synthetic checks of inclusive windows, missing changes, and patient weights."""

import unittest

import numpy as np
import pandas as pd

from .analyze import window_statistics, summarize, HOURS


class WindowVariabilityTests(unittest.TestCase):
    def test_nested_continuous_windows_and_nonmonotone_net_change(self):
        center = 200000.
        times = center + np.array([-25,-13,-7,-5,0,5,7,13,25])*3600
        values = np.array([.5,1.,1.5,2.,2.5,1.3,2.2,3.,4.])
        rows = [window_statistics(times,values,center,center+60,0,500000,h,2.,"high",1.) for h in HOURS]
        self.assertEqual([row["n_lab_events"] for row in rows],[3,5,7])
        self.assertLess(rows[0]["signed_change"],0)
        self.assertGreater(rows[1]["signed_change"],0)
        self.assertTrue(all(a["value_range"]<=b["value_range"] for a,b in zip(rows,rows[1:])))
        self.assertEqual(rows[0]["observed_span_h"],10)

    def test_exact_window_boundaries_and_admission_clipping(self):
        start=100000.;end=start+60
        times=np.array([start-6*3600-1,start-6*3600,end+6*3600,end+6*3600+1])
        values=np.arange(4,dtype=float)
        row=window_statistics(times,values,start,end,0,300000,6,2,"high",1)
        self.assertEqual(row["n_lab_events"],2)
        self.assertEqual((row["first_value"],row["last_value"]),(1,2))
        row=window_statistics(times,values,start,end,start-3600,end+3600,6,2,"high",1)
        self.assertEqual(row["n_lab_events"],0)
        self.assertEqual(row["window_clipped_by_admission"],1)
        self.assertEqual(row["window_clipped_by_discharge"],1)

    def test_single_measurement_does_not_imply_no_change(self):
        row=window_statistics(np.array([100000.]),np.array([3.]),100000,100060,0,300000,6,2,"high",1)
        self.assertEqual(row["n_lab_events"],1)
        self.assertEqual(row["nearest_value"],3)
        for name in ("signed_change","value_range","sample_sd","threshold_crossing"):
            self.assertTrue(np.isnan(row[name]))

    def test_threshold_equality_and_nearest_tie_break(self):
        times=np.array([99980.,100080.])
        high=window_statistics(times,np.array([2.,2.1]),100000,100060,0,300000,6,2,"high",1)
        low=window_statistics(times,np.array([94.,93.9]),100000,100060,0,300000,6,94,"low",2)
        self.assertEqual(high["threshold_crossing"],1)
        self.assertEqual(low["threshold_crossing"],1)
        self.assertEqual(high["nearest_value"],2)
        self.assertAlmostEqual(high["nearest_match_delta_h"],20/3600)

    def test_patient_not_video_weighting_in_paired_comparison(self):
        rows=[]
        for patient,video,magnitude in (("a","a1",1.),("a","a2",1.),("a","a3",1.),("b","b1",9.)):
            for hours in HOURS:
                row=window_statistics(np.array([100000.,100060.]),np.array([1.,1.+magnitude]),100020,100040,0,300000,hours,2,"high",1)
                rows.append({**row,"target":"lactate_high","hospital_id":patient,"video_id":video,"window_half_width_h":hours})
        patients,summary,coverage=summarize(pd.DataFrame(rows))
        selected=summary.loc[summary.cohort.eq("common_6h_videos") & summary.metric.eq("value_range")]
        self.assertTrue(selected.estimate.eq(5).all())
        self.assertTrue(selected.patients.eq(2).all())
        self.assertTrue(selected.videos.eq(4).all())
        selected=summary.loc[summary.cohort.eq("common_6h_videos") & summary.metric.eq("threshold_crossing")]
        self.assertTrue(selected.estimate.eq(.5).all())


if __name__=="__main__":unittest.main()
