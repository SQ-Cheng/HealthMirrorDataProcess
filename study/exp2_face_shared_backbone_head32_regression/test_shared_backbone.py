"""Verify missing-label handling, equal-task loss, batch coverage and sharing."""

import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from . import config
from .shared_backbone import SharedBackboneModel, SharedLabBatchSampler, task_balanced_loss
from .run_all import build_union, macro_scaled_mae, execution_model, clear_outputs


class SharedBackboneTests(unittest.TestCase):
    def test_union_retains_partial_labels_and_correct_target_columns(self):
        records = {}
        for column, target in enumerate(config.ALL_REGRESSION_TARGETS):
            ids = ["v0", "v1"] if column == 0 else ["v1", "v2"] if column == 1 else ["v1"]
            records[target] = pd.DataFrame({"hospital_id": ["p" + video[1:] for video in ids], "video_id": ids,
                "mirror": "m", "lab_patient_id": 1, "split": "train", "robust_scaled_raw_value": column + 1.,
                "clinical_event_id": [f"{target}@{video}" for video in ids]})
        videos, labels, masks, events = build_union(records)
        self.assertEqual(videos.video_id.tolist(), ["v0", "v1", "v2"])
        np.testing.assert_array_equal(masks.sum(axis=1), [1, 10, 1])
        np.testing.assert_array_equal(masks.sum(axis=0), [2, 2] + [1] * 8)
        np.testing.assert_array_equal(labels[1], np.arange(1, 11))
        self.assertEqual(len(events[1]), 10)
        records[config.ALL_REGRESSION_TARGETS[0]] = pd.concat([records[config.ALL_REGRESSION_TARGETS[0]]] * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_union(records)

    def test_overwrite_keeps_live_lock_and_other_experiments(self):
        from . import run_all
        with tempfile.TemporaryDirectory() as name:
            here = Path(name)
            output = here / "outputs"; output.mkdir()
            lock = output / ".queue.lock"; lock.write_text("lock")
            (output / "runs").mkdir(); (output / "runs/model.pt").write_text("old")
            (output / "model.pt").write_text("old")
            other = here / "other_model.pt"; other.write_text("keep")
            with patch.object(config, "EXP_DIR", here), patch.object(run_all, "OUTPUT", output):
                clear_outputs()
            self.assertEqual(list(output.iterdir()), [lock])
            self.assertEqual(other.read_text(), "keep")

    def test_training_settings_align_main_regression(self):
        from study.exp2_face_pretrained_head32_regression import config as main
        for name in ("HEAD_LEARNING_RATE", "FINETUNE_LEARNING_RATE", "HEAD_MAX_EPOCHS", "FINETUNE_MAX_EPOCHS",
                     "HEAD_PATIENCE", "FINETUNE_PATIENCE", "MIN_LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM",
                     "SMOOTH_L1_BETA", "TRAIN_NUM_WORKERS", "EVAL_NUM_WORKERS", "PREFETCH_FACTOR", "VIEW_NAMES"):
            self.assertEqual(getattr(config, name), getattr(main, name))

    def test_missing_nan_targets_have_zero_gradient(self):
        prediction = torch.zeros(3, 2, requires_grad=True)
        mask = torch.tensor([[True, False], [False, True], [True, True]])
        labels = torch.tensor([[1., float("nan")], [float("nan"), 2.], [3., 4.]])
        loss = task_balanced_loss(prediction, labels, mask, torch.ones(2))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[~mask], torch.zeros(2)))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_sparse_and_dense_tasks_have_equal_aggregate_weight(self):
        prediction = torch.zeros(10, 2, requires_grad=True)
        labels = torch.tensor([[1., 3.]] * 10)
        mask = torch.zeros(10, 2, dtype=torch.bool); mask[:, 0] = True; mask[:2, 1] = True
        weights = 10 / (2 * mask.sum(dim=0))
        loss = task_balanced_loss(prediction, labels, mask, weights)
        self.assertAlmostEqual(loss.item(), (.75 + 2.75) / 2)
        loss.backward()
        torch.testing.assert_close(prediction.grad.sum(dim=0), torch.tensor([-.5, -.5]))

    def test_batch_never_repeats_a_target_assay_and_retains_all_views(self):
        events = [{"hb|p0@1", "cr|p0@2"}, {"hb|p0@1", "cr|p0@3"}, {"hb|p1@1"}]
        sampler = SharedLabBatchSampler(events)
        self.assertEqual(len(sampler), len(list(sampler)))
        self.assertEqual(sorted(np.concatenate(sampler.batches).tolist()), list(range(300)))
        for batch in sampler:
            used = set()
            for group in np.asarray(batch).reshape(-1, 20):
                self.assertEqual(len(set(group // 100)), 1)
                self.assertEqual(len(set(group % 5)), 1)
                selected = events[group[0] // 100]
                self.assertFalse(used & selected); used.update(selected)
        first = [list(batch) for batch in sampler]
        sampler.set_epoch(1)
        self.assertNotEqual(first, sampler.batches)
        self.assertEqual(sorted(np.concatenate(sampler.batches).tolist()), list(range(300)))

    def test_one_encoder_ten_separate_heads(self):
        torch.set_num_threads(1)
        model = SharedBackboneModel(config.ALL_REGRESSION_TARGETS)
        self.assertEqual(sum(p.numel() for p in model.features.parameters()), 4007548)
        self.assertEqual(sum(p.numel() for p in model.heads.parameters()), 410890)
        model.freeze_encoder()
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.heads.parameters()))
        self.assertEqual(len({id(head[0].weight) for head in model.heads.values()}), 10)
        model.eval()
        with torch.no_grad(): outputs = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(outputs.shape, (2, 10))
        self.assertTrue(torch.isfinite(outputs).all())

    def test_cross_task_patient_split_conflict_is_rejected(self):
        records = {}
        for index, target in enumerate(config.ALL_REGRESSION_TARGETS):
            records[target] = pd.DataFrame({"hospital_id": ["p"], "video_id": ["v"], "mirror": ["m"],
                                            "lab_patient_id": [1], "split": ["train" if index == 0 else "test"]})
        with self.assertRaisesRegex(ValueError, "leak"):
            build_union(records)

    def test_joint_selection_uses_scale_normalized_video_mae(self):
        metrics = {target: {"mae": index + 1.} for index, target in enumerate(config.ALL_REGRESSION_TARGETS)}
        scalers = {target: {"iqr": 2 * (index + 1.)} for index, target in enumerate(config.ALL_REGRESSION_TARGETS)}
        self.assertEqual(macro_scaled_mae(metrics, scalers), .5)

    def test_compile_does_not_enable_cuda_graphs(self):
        model = torch.nn.Linear(3, 1)
        with patch.object(torch, "compile", return_value=model) as compile_model:
            compiled, backend = execution_model(model)
        compile_model.assert_called_once_with(model, dynamic=True, options={"triton.cudagraphs": False})
        self.assertIs(compiled, model)
        self.assertEqual(backend, "inductor:default_dynamic_no_cudagraphs")


if __name__ == "__main__":
    unittest.main()
