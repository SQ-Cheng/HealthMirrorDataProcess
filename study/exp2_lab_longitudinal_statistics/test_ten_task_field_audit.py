"""Focused tests for the read-only field audit and its shadow matching."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from .audit_ten_task_fields import alias_decision, compare_matches, rematch, select_rows


def rows(items, values, units, specimens=None):
    n = len(items)
    numeric = pd.to_numeric(pd.Series(values).str.lstrip("?"), errors="coerce")
    return pd.DataFrame({
        "检验项名称": items, "检验值(文本)": values, "单位": units,
        "标本名称": specimens or ["血"] * n, "hospital_id": ["123"] * n,
        "timestamp_unix": [100.] * n, "numeric_value": numeric,
        "exact_value": numeric.where(~pd.Series(values).str.startswith("?")),
    })


class FieldAuditTests(unittest.TestCase):
    def test_hb_conversion_and_nonexact_flag(self):
        data = rows(["*血红蛋白", "血红蛋白", "血红蛋白"], ["100", "10", "?10"], ["g/l", "g/dL", "g/dL"])
        result, _ = select_rows(data, "hemoglobin")
        self.assertEqual(result.value.tolist(), [100., 100., 100.])
        self.assertTrue(result.accepted.all())
        self.assertEqual(result.nonexact_numeric.tolist(), [False, False, True])

    def test_ratio_whitelist_units_and_venous(self):
        data = rows(["动脉血氧分压与肺泡内氧分压之比", "动脉氧分压与肺泡氧分压之比",
                     "动脉-肺泡氧分压比值", "动脉氧分压与肺泡氧分压之比"],
                    ["0.4", "40", "0.4", "40"], ["", "%", "", "%"], ["血", "血", "全血", "静脉血"])
        result, _ = select_rows(data, "aa_po2_ratio")
        self.assertEqual(result.value.tolist(), [40., 40., 40.])
        self.assertEqual(result.accepted.tolist(), [True, True, False])
        self.assertEqual(alias_decision("aa_po2_ratio", "动脉-肺泡氧分压比值")[0], "omitted_ratio_alias")

    def test_direct_fields_specimen_and_missing_unit(self):
        data = rows(["eGFR(CKD-EPI 肌酐)"]*3, ["90"]*3,
                    ["mL/min/1.73㎡", "", "mL/min/1.73㎡"], ["血", "血", "尿"])
        result, _ = select_rows(data, "egfr_creatinine")
        self.assertEqual(result.accepted.tolist(), [True, False, False])

    def test_numeric_disagreement_is_detected(self):
        data = rows(["乳酸"], ["1e1"], ["mmol/L"])
        data["numeric_value"], data["exact_value"] = 1., 10.
        result, _ = select_rows(data, "lactate")
        self.assertTrue(result.numeric_parser_disagreement.iloc[0])

    def test_same_admission_and_interval_boundary(self):
        pool = pd.DataFrame({"hospital_id": ["123"]*3, "timestamp_unix": [50., 120., 300.], "value": [1., 2., 3.]})
        videos = pd.DataFrame([vars(SimpleNamespace(hospital_id="123", video_id="v", admission_unix=100.,
                                                  discharge_unix=200., capture_start_unix=110., capture_end_unix=115.))])
        matched = rematch(pool, videos)
        self.assertEqual(matched.raw_value.tolist(), [2.])
        impact, changed = compare_matches(matched, matched, "lactate_high", "same")
        self.assertEqual(len(changed), 0)
        self.assertEqual(impact["added_videos"], 0)
        self.assertEqual(impact["changed_time_videos"], 0)
        self.assertTrue(np.isclose(matched.match_delta_h.iloc[0], 5/3600))


if __name__ == "__main__":
    unittest.main()
