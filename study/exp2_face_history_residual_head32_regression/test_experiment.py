"""Focused patient isolation, scaling and residual composition tests."""

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from study.exp2_face_history_head32_regression.history_data import HistoryFeatureStore
from study.exp2_face_history_head32_regression.scaling import fit_robust_target_scaler
from study.exp2_history_only_head32_regression.models import HistoryOnlyRegressor, parameter_counts
from .data import ResidualScaler, crossfit_splits, residual_records, rescale_history, validate_patient_split
from . import run


class ExperimentTests(unittest.TestCase):
    def records(self):
        return pd.DataFrame({"hospital_id": [str(i//2) for i in range(90)],
                             "video_id": [f"v{i}" for i in range(90)],
                             "raw_value": np.linspace(50, 150, 90), "score_threshold": 130.,
                             "split": ["train"]*60+["val"]*14+["test"]*16})

    def test_patient_oof_no_outer_holdout_exposure(self):
        records = self.records()
        validate_patient_split(records)
        outer_holdout = set(records.loc[records.split.ne("train"), "hospital_id"])
        held_out_ids = []
        for fold in crossfit_splits(records, 42):
            self.assertFalse(set(fold.hospital_id) & outer_holdout)
            self.assertTrue(fold.groupby("hospital_id").split.nunique().eq(1).all())
            held_out_ids.extend(fold.loc[fold.split.eq("test"), "video_id"])
        self.assertEqual(set(held_out_ids), set(records.loc[records.split.eq("train"), "video_id"]))
        self.assertEqual(len(held_out_ids), len(set(held_out_ids)))

    def test_residual_exact_composition_and_scale(self):
        records = self.records()
        scaler = fit_robust_target_scaler("hemoglobin_low", records, "g/L")
        baseline = records[["hospital_id", "video_id"]].copy()
        baseline["history_pred_raw"] = 95.
        baseline["prediction_source"] = np.where(records.split.eq("train"), "patient_oof", "outer_train_fit")
        residual = residual_records(records, baseline, scaler)
        np.testing.assert_allclose(residual.raw_value+residual.history_pred_raw, records.raw_value)
        np.testing.assert_allclose(residual.score_threshold+residual.history_pred_raw, records.score_threshold)
        residual_scaler = ResidualScaler(scaler.target, scaler.unit, scaler.iqr, scaler.to_dict())
        np.testing.assert_allclose(residual_scaler.inverse_transform(residual.robust_scaled_raw_value), residual.raw_value)
        baseline["prediction_source"] = "outer_train_fit"
        with self.assertRaises(ValueError):
            residual_records(records, baseline, scaler)

    def test_fold_history_scaling_and_time_preservation(self):
        records = self.records()
        full = fit_robust_target_scaler("hemoglobin_low", records, "g/L")
        inner = records.copy()
        inner.loc[inner.index > 30, "split"] = "test"
        fold = fit_robust_target_scaler("hemoglobin_low", inner, "g/L")
        raw = np.array([40., 80., 100.])
        original = HistoryFeatureStore(np.array(["v0"]), np.array([0, 3]),
            np.column_stack([full.transform(raw), [-3., -2., -1.]]).astype(np.float32), 3)
        changed = rescale_history(original, fold, raw)
        np.testing.assert_allclose(changed.features[:, 0], fold.transform(raw), atol=1e-6)
        np.testing.assert_array_equal(changed.features[:, 1], original.features[:, 1])
        np.testing.assert_array_equal(original.features[:, 0], full.transform(raw).astype(np.float32))

    def test_history_parameter_budget(self):
        self.assertEqual(parameter_counts(HistoryOnlyRegressor()), {"history_encoder": 352, "head": 641, "total": 993})

    def test_compile_disables_cuda_graphs(self):
        model = object()
        with patch.object(run.face_train, "TORCH_COMPILE_ENABLED", True), patch.object(run.torch, "compile") as compile_model:
            execution, backend = run.execution_model(model, "efficientnet_b0", "hemoglobin_low", "finetune", SimpleNamespace(type="cuda"))
        compile_model.assert_called_once_with(model, dynamic=True, options={"triton.cudagraphs": False})
        self.assertEqual(execution, compile_model.return_value)
        self.assertEqual(backend, "inductor:default_dynamic_no_cudagraphs")

    def test_task_cleanup_also_runs_on_failure(self):
        with patch.object(run, "release_gpu_cache") as clean, patch.object(run, "_face_job_impl", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                run.face_job({})
        self.assertEqual(clean.call_count, 2)

    def test_overwrite_preserves_prepared_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("task_records", "history_records", "source_data", "runs", "residual_records", "figures", "tables"):
                (output / name).mkdir()
                (output / name / "sample.txt").write_text("unchanged")
            for name in ("protocol.json", "target_scalers.json", "dataset_counts.csv", "history_run_index.csv", "COMPLETED.json"):
                (output / name).write_text("unchanged")
            run.clear_training_results(output)
            for name in ("task_records", "history_records", "source_data"):
                self.assertEqual((output / name / "sample.txt").read_text(), "unchanged")
            for name in ("protocol.json", "target_scalers.json", "dataset_counts.csv"):
                self.assertEqual((output / name).read_text(), "unchanged")
            self.assertFalse((output / "runs").exists())
            self.assertFalse((output / "history_run_index.csv").exists())


if __name__ == "__main__":
    unittest.main()
