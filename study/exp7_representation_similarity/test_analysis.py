"""Small deterministic checks for the held-out association analysis."""

import unittest

import numpy as np
import pandas as pd

from .analyze import _association, _fdr_bh, _one_per_patient


class RepresentationAnalysisTest(unittest.TestCase):
    def test_patient_selection_does_not_use_labels(self):
        records = pd.DataFrame({
            "hospital_id": ["a", "a", "b", "b"],
            "video_id": ["a1", "a2", "b1", "b2"],
            "raw_value": [1.0, 2.0, 3.0, 4.0],
        })
        first = _one_per_patient(records, "video_id")
        changed = records.assign(raw_value=[20.0, -20.0, 100.0, -100.0])
        second = _one_per_patient(changed, "video_id")
        self.assertEqual(first.video_id.tolist(), second.video_id.tolist())
        self.assertEqual(len(first), 2)

    def test_similarity_detects_label_order(self):
        labels = np.linspace(-1.0, 1.0, 24)
        features = np.column_stack((np.cos(labels), np.sin(labels)))
        strata = np.repeat(np.arange(4), 6)
        result = _association(features, labels, permutations=199, seed=7, strata=strata)
        self.assertGreater(result["rho"], 0.9)
        self.assertLessEqual(result["permutation_p_one_sided"], 0.01)
        self.assertLessEqual(result["interval_stratified_p_one_sided"], 0.01)
        self.assertEqual(result["pair_count"], 24 * 23 // 2)

    def test_bh(self):
        np.testing.assert_allclose(_fdr_bh([0.01, 0.03, 0.4]), [0.03, 0.045, 0.4])


if __name__ == "__main__":
    unittest.main()
