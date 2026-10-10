"""Small source, temporal-boundary and patient-identity regression tests."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from study.exp2_face_history_head32_regression.source_data import _binary_label, _canonical_values
from .run_preoperative_ablation import select_measurement, extend_patient_split


class PreoperativeTests(unittest.TestCase):
    def test_no_preoperative_distance_limit(self):
        chosen, pre = select_measurement([(10, 2), (1000000, 9)], 500000, 500001, 0, 2000000, 600000)
        self.assertTrue(pre)
        self.assertEqual(chosen["timestamp_unix"], 10)
        self.assertGreater(chosen["delta_h"], 24)

    def test_no_intra_or_postoperative_label(self):
        chosen, pre = select_measurement([(90, 3), (100, 4), (105, 5)], 97, 99, 0, 200, 100)
        self.assertTrue(pre)
        self.assertEqual(chosen["value"], 3)
        self.assertIsNone(select_measurement([(100, 4), (105, 5)], 97, 99, 0, 200, 100)[0])

    def test_same_admission_only(self):
        chosen, _ = select_measurement([(9, 4), (20, 5), (110, 6)], 15, 16, 10, 100, 90)
        self.assertEqual(chosen["timestamp_unix"], 20)

    def test_non_preoperative_rule_unchanged(self):
        events = [(0, 4)]
        self.assertIsNone(select_measurement(events, 100000, 100001, 0, 200000, None)[0])
        self.assertIsNone(select_measurement(events, 100000, 100001, 0, 200000, 10)[0])
        self.assertEqual(select_measurement(events, 86400, 86401, 0, 200000, None)[0]["value"], 4)

    def test_tie_break_earlier_report(self):
        selected, _ = select_measurement([(90, 1), (112, 2)], 100, 102, 0, 200, 190)
        self.assertEqual(selected["timestamp_unix"], 90)

    def test_unit_validation_no_missing_unit_guess(self):
        values, valid = _canonical_values("egfr_creatinine", pd.Series([""] * 4),
                                         pd.Series([70] * 4), pd.Series(["mL/min/1.73㎡", "mL/(min/1.73㎡)", "", "mg/dL"]))
        self.assertEqual(valid.tolist(), [True, True, False, False])
        self.assertEqual(values.tolist(), [70] * 4)
        _, valid = _canonical_values("hematocrit", pd.Series(["红细胞压积"] * 2),
                                    pd.Series([30, 30]), pd.Series(["%", ""]))
        self.assertEqual(valid.tolist(), [True, False])

    def test_sex_specific_hct_auxiliary_labels(self):
        self.assertEqual(_binary_label("hematocrit_low", 35, "男"), 1)
        self.assertEqual(_binary_label("hematocrit_low", 35, "女"), 0)
        self.assertEqual(_binary_label("egfr_low", 60, "男"), 0)

    def test_existing_patients_never_move(self):
        reference = pd.DataFrame({"hospital_id": ["001", "002", "003"], "split": ["train", "val", "test"]})
        records = pd.DataFrame({"hospital_id": ["003", "001", "004", "005", "006", "007"]})
        first = extend_patient_split(records, reference, "egfr_low")
        second = extend_patient_split(records, reference, "egfr_low")
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.split.iloc[:2].tolist(), ["test", "train"])
        self.assertFalse(first.split.isna().any())

    def test_ten_target_figures_and_comparison(self):
        from . import config, plot_results
        from .plot_preoperative_comparison import plot_comparison
        from .train import _regression_metrics
        with tempfile.TemporaryDirectory(prefix="preoperative_plot_test_") as temporary:
            roots = [Path(temporary) / name for name in ("main", "ablation")]
            for variant, root in enumerate(roots):
                metrics, histories, index = [], [], []
                root.mkdir()
                for target in config.ALL_REGRESSION_TARGETS:
                    run = root / f"runs/efficientnet_b0/{target}"; run.mkdir(parents=True)
                    true = np.array([40., 50., 70.])
                    predicted = true + np.array([1., -2., 3.]) * (variant + 1)
                    definition = config.SCORE_DEFINITIONS[target]
                    threshold = definition["threshold"]
                    if isinstance(threshold, dict): threshold = threshold["male"]
                    prediction = []
                    for split in ("train", "val", "test"):
                        values = _regression_metrics(true, predicted, [threshold] * 3, definition["direction"])
                        metrics.append({"target": target, "architecture": "efficientnet_b0", "split": split, **values})
                        for row in range(3):
                            prediction.append({"hospital_id": f"{split}_{row}", "video_id": f"{split}_{row}",
                                               "split": split, "target": target, "y_true": true[row],
                                               "y_pred": predicted[row], "frame_count": 20, "binary_label": row % 2,
                                               "score_threshold": threshold})
                    pd.DataFrame(prediction).to_csv(run / "video_predictions.csv", index=False)
                    index.append({"architecture": "efficientnet_b0", "target": target, "status": "ok"})
                    for epoch in range(1, 5):
                        histories.append({"target": target, "architecture": "efficientnet_b0", "global_epoch": epoch,
                                          "stage": "head" if epoch < 3 else "finetune", "train_eval_loss": 1 / epoch,
                                          "val_loss": 1.1 / epoch, "val_mae": 3 / epoch, "val_pearson_r": .1 * epoch})
                pd.DataFrame(metrics).to_csv(root / "metrics_all.csv", index=False)
                pd.DataFrame(histories).to_csv(root / "history_all.csv", index=False)
                pd.DataFrame(index).to_csv(root / "run_index.csv", index=False)
            with patch.object(plot_results, "OUTPUT_DIR", plot_results.OUTPUT_DIR), \
                 patch.object(plot_results, "FIGURE_DIR", plot_results.FIGURE_DIR), \
                 patch.object(plot_results, "TASKS", plot_results.TASKS), \
                 patch.object(plot_results, "ARCHITECTURES", plot_results.ARCHITECTURES):
                plot_results.main(roots[1])
            plot_comparison(*roots)
            self.assertEqual(len(list((roots[1] / "figures").glob("*.png"))), 6)
            table = pd.read_csv(roots[1] / "tables/shared_test_unchanged_label_comparison.csv")
            self.assertEqual(set(table.target), set(config.ALL_REGRESSION_TARGETS))
            self.assertTrue(table.n.eq(3).all())


if __name__ == "__main__":
    unittest.main()
