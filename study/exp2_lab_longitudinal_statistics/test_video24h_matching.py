"""Exact-value parsing, semantic separation and same-admission nearest counts."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .video24h_matching import parse_times, exact_numeric, specimen_group, match_events, nearest_sorted, family_name
from study.exp2_face_history_head32_regression.source_data import _nearest_measurement
from .run_analysis import _harmonize_analytes


class MatchingTests(unittest.TestCase):
    def test_related_words_do_not_merge_different_measurements(self):
        self.assertNotEqual(family_name("乳酸浓度"), family_name("乳酸脱氢酶"))
        self.assertEqual(family_name("尿酸碱度"), "酸碱与通气")
        self.assertNotEqual(family_name("尿酸碱度"), family_name("尿酸"))

    def test_search_neighbors_exactly_match_full_nearest_scan(self):
        rng = np.random.default_rng(13)
        for _ in range(300):
            times = np.sort(rng.choice(np.arange(0, 300000), size=100, replace=False)).astype(float)
            values = rng.normal(size=len(times))
            start = float(rng.integers(50000, 250000))
            video = SimpleNamespace(capture_start_unix=start, capture_end_unix=start + rng.integers(1, 4000),
                                    admission_unix=start - 10000, discharge_unix=start + 15000)
            eligible = (times >= video.admission_unix) & (times <= video.discharge_unix)
            expected = _nearest_measurement(list(zip(times[eligible], values[eligible])), video.capture_start_unix, video.capture_end_unix, 24)
            self.assertEqual(nearest_sorted(times, values, video), expected)

    def test_numeric_does_not_extract_ranges_ordinals_or_censored_values(self):
        values = exact_numeric(pd.Series(["1.2", "1,234.5", "1e-3", "−2", "<0.1", "2+", "0-5", "1:80", "1,2", "阴性", "", "inf"]))
        np.testing.assert_allclose(values.iloc[:4], [1.2, 1234.5, .001, -2])
        self.assertTrue(values.iloc[4:].isna().all())

    def test_mixed_dates_and_timezone(self):
        result = parse_times(pd.Series(["2025-01-02 12:00:00", "2025/1/2 12:00", "2025-01-02T04:00:00Z", "bad"]))
        np.testing.assert_array_equal(result.iloc[:3], np.repeat(result.iloc[0], 3))
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_distinct_gas_specimens_and_noninterchangeable_items(self):
        self.assertEqual(specimen_group("血清", "肌酐(Cr)测定"), "血液")
        self.assertNotEqual(specimen_group("尿", "肌酐(Cr)测定"), "血液")
        self.assertNotEqual(specimen_group("动脉血即刻", "氧分压"), specimen_group("静脉血", "氧分压"))
        raw = pd.DataFrame({"item_name": ["氧分压", "患者体温下氧分压", "eGFR(CKD-EPI 肌酐)", "eGFR(CKD-EPI 胱抑素C)"],
                            "unit": ["mmhg", "mmhg", "ml/min/1.73m^2", "ml/min/1.73m^2"],
                            "numeric_value": [80., 75., 70., 60.], "hospital_id": "p", "标本名称": "血"})
        harmonized, _ = _harmonize_analytes(raw)
        self.assertEqual(harmonized.item_name.nunique(), 4)

    def test_unit_conversion_and_aliases(self):
        raw = pd.DataFrame({"item_name": ["*血红蛋白", "血红蛋白", "葡萄糖浓度", "葡萄糖浓度"],
                            "unit": ["g/l", "g/dl", "mmol/l", "mg/dl"], "numeric_value": [120., 12., 5., 90.091],
                            "hospital_id": "p", "标本名称": "血"})
        harmonized, _ = _harmonize_analytes(raw)
        np.testing.assert_allclose(harmonized.numeric_value, [120., 120., 5., 5.])
        self.assertEqual(harmonized.item_name.nunique(), 2)

    def test_episode_boundaries_tie_breaking_and_one_match_per_video(self):
        videos = pd.DataFrame([{"hospital_id": "p", "video_id": "v", "mirror": "m", "capture_start_unix": 100000.,
                                "capture_end_unix": 100010., "admission_unix": 10000., "discharge_unix": 200000.,
                                "native224_20frame_eligible": True}])
        labs = pd.DataFrame({"canonical_id": ["A", "A", "A", "B", "C", "D"], "hospital_id": ["p"] * 5 + ["other"],
                             "report_unix": [99990., 99990., 100020., 186410., 186411., 100000.],
                             "numeric_value": [2., 4., 8., 10., 11., 12.], "numeric_valid": True, "any_valid": True})
        matched = match_events(labs, videos, "canonical_id")
        self.assertEqual(matched.canonical_id.tolist(), ["A", "B"])
        self.assertEqual(matched.raw_value.tolist(), [3., 10.])
        self.assertEqual(matched.match_delta_h.iloc[1], 24.)
        videos.loc[0, "admission_unix"] = 100000.
        matched = match_events(labs, videos, "canonical_id")
        self.assertEqual(matched.loc[matched.canonical_id.eq("A"), "raw_value"].iloc[0], 8.)


if __name__ == "__main__":
    unittest.main()
