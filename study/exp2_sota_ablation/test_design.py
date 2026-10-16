"""CPU/synthetic tests only: no clinical optimization or GPU training."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from study.exp2_face_dinov3_farl_regression.models import ScalarGatedHead
from . import config
from .data import fit_transform, transform, inverse, train_weights, PatientEventBatchSampler
from .models import Fusion
from .online import select_tail, OnlineIR
from .train import weighted_objective, schedule_lr


class DesignTests(unittest.TestCase):
    def test_all_seven_heads_have_gradients_and_head32(self):
        for name in config.VARIANTS:
            settings = config.get_variant(name)
            model = Fusion(settings).eval()
            features = torch.randn(4, 2560, requires_grad=True)
            output = model(features)
            self.assertEqual(tuple(output.shape), (4,))
            output.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
            self.assertTrue(torch.isfinite(features.grad).all())
            head = model.head if settings["model"] == "hierarchical" else model.network.head
            self.assertEqual(head[0].out_features, 32)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 167172 if settings["model"] == "hierarchical" else 166530)

    def test_warmstart_scaler_conversion_preserves_rgb_expert_raw_output(self):
        old = ScalarGatedHead(dino_features=768).eval()
        model = Fusion(config.get_variant("a03_hierarchical_rgb_warmstart")).eval()
        checkpoint = {"state_dict": old.state_dict(), "target_scaler": {"unit": "g/L", "median": 110., "iqr": 30.}}
        model.warm_start_rgb(checkpoint, {"unit": "g/L", "median": 100., "iqr": 25.})
        inputs = torch.randn(8, 1280)
        with torch.no_grad():
            expected = old(inputs)*30+110
            actual = model.head(model.rgb_expert(inputs))*25+100
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
        self.assertNotEqual(model.rgb_expert.dino_project[0].weight.data_ptr(), model.ir_expert.dino_project[0].weight.data_ptr())
        torch.testing.assert_close(model.modality_gate(torch.randn(8, 128)).sigmoid(), torch.full((8, 1), .95))

    def test_train_only_transform_and_log_round_trip(self):
        records = pd.DataFrame({"split": ["train"]*4+["val", "test"], "raw_value": [1., 2., 4., 8., 100., 300.]})
        settings = config.get_variant("a04_log_density_rank")
        scaler = fit_transform(records, "lactate_high", settings)
        changed = records.copy()
        changed.loc[changed.split.ne("train"), "raw_value"] *= 1e6
        self.assertEqual(scaler, fit_transform(changed, "lactate_high", settings))
        np.testing.assert_allclose(inverse(transform(records.raw_value, scaler), scaler), records.raw_value, rtol=1e-6)
        self.assertTrue(scaler["log1p"])
        self.assertFalse(fit_transform(records, "hemoglobin_low", settings)["log1p"])

    def test_patient_event_weights(self):
        records = pd.DataFrame({"split": "train", "hospital_id": ["a", "a", "a", "b"],
            "clinical_event_id": ["e1", "e1", "e2", "e3"], "video_id": ["v1", "v2", "v3", "v4"],
            "raw_value": [1., 1., 2., 3.], "match_delta_h": [0., 0., 24., 6.]})
        weights, _ = train_weights(records, config.get_variant("a02_video_patient_balance"))
        np.testing.assert_allclose(weights[:3].sum(), weights[3])
        np.testing.assert_allclose(weights[:2].sum(), weights[2])
        settings = config.get_variant("a06_time_weighted")
        settings["balance"] = False
        weights, _ = train_weights(records, settings)
        self.assertGreater(weights[0], weights[3])
        self.assertGreater(weights[3], weights[2])
        records.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            train_weights(records, settings)

    def test_patient_diverse_sampler_exhaustive_views_no_duplicates(self):
        records = pd.DataFrame({"hospital_id": ["a"]*6+["b"]*3+["c"]*3,
                                "clinical_event_id": ["e1"]*3+["e2"]*3+["e3"]*3+["e4"]*3})
        dataset = SimpleNamespace(video_records=records, views=config.reference.VIEWS, expand_all_views=False,
                                  frame_video_rows=np.repeat(np.arange(12), 20))
        sampler = PatientEventBatchSampler(dataset, 240)
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(12*100))
        for batch in batches:
            grouped = np.array(batch).reshape(-1, 20)
            selected = records.iloc[grouped[:, 0]//100]
            self.assertEqual(selected.hospital_id.nunique(), len(grouped))
            self.assertEqual(selected.clinical_event_id.nunique(), len(grouped))
            self.assertTrue(all(len(set(group % 5)) == 1 for group in grouped))

    def test_frame_vs_video_objective_and_ranking(self):
        prediction = torch.tensor([-1., 1.]*10+[0., 2.]*10, requires_grad=True)
        labels = torch.tensor([0.]*20+[1.]*20)
        rows = torch.tensor([0]*20+[1]*20)
        settings = config.get_variant("a02_video_patient_balance")
        loss = weighted_objective(prediction, labels, rows, torch.ones(2), settings)
        self.assertAlmostEqual(float(loss), 0., places=6)
        settings["loss"] = "frame"
        self.assertGreater(float(weighted_objective(prediction, labels, rows, torch.ones(2), settings)), .1)
        settings["loss"], settings["rank"] = "video_view", .05
        right = weighted_objective(prediction, labels, rows, torch.ones(2), settings)
        wrong = weighted_objective(-prediction, labels, rows, torch.ones(2), settings)
        self.assertGreater(float(wrong), float(right))
        wrong.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_warmup_and_cosine_endpoints(self):
        parameter = nn.Parameter(torch.tensor(1.))
        optimizer = torch.optim.SGD([parameter], lr=1e-4)
        values = []
        for epoch in range(1, 41):
            schedule_lr(optimizer, epoch, 40, 2, [1e-4], [1e-6])
            values.append(optimizer.param_groups[0]["lr"])
        np.testing.assert_allclose(values[:2], [5e-5, 1e-4])
        self.assertAlmostEqual(values[-1], 1e-6)

    def test_ir_tail_selection_and_small_checkpoint_round_trip(self):
        dino = nn.Module()
        dino.blocks, dino.norm = nn.ModuleList([nn.Linear(3, 3) for _ in range(3)]), nn.LayerNorm(3)
        farl = nn.Module()
        farl.transformer = nn.Module()
        farl.transformer.resblocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(3)])
        farl.ln_post, farl.proj = nn.LayerNorm(3), nn.Parameter(torch.randn(3, 2))
        select_tail(dino, farl)
        self.assertFalse(any(p.requires_grad for p in dino.blocks[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in farl.transformer.resblocks[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in dino.blocks[-1].parameters()))
        self.assertTrue(farl.proj.requires_grad)
        model = OnlineIR.__new__(OnlineIR)
        nn.Module.__init__(model)
        model.dino, model.farl = dino, farl
        state = model.tail_state()
        self.assertTrue(all(not key.startswith(("dino.blocks.0", "farl.transformer.resblocks.0")) for key in state))
        with torch.no_grad():
            farl.proj.zero_()
        model.load_tail(state)
        torch.testing.assert_close(farl.proj, state["farl.proj"])

    def test_default_cli_does_not_train(self):
        from .run import main
        output = io.StringIO()
        with patch("sys.argv", ["run"]), patch("study.exp2_sota_ablation.run.validate_sources") as validate, contextlib.redirect_stdout(output):
            main()
        self.assertFalse(json.loads(output.getvalue())["training_started"])
        validate.assert_not_called()

    def test_queue_failure_stops_subsequent_variants(self):
        from .run import main
        names = list(config.VARIANTS)[:2]
        with tempfile.TemporaryDirectory() as name, patch.object(config, "OUTPUT", Path(name)), patch("sys.argv", ["run", "--run", "--variants", *names]), patch(
                "study.exp2_sota_ablation.run.validate_sources", return_value=(None, None, {})), patch(
                "torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=8), patch(
                "study.exp2_sota_ablation.run.execute", side_effect=RuntimeError("synthetic failure")) as execute:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                main()
            self.assertEqual(execute.call_count, 1)
            self.assertFalse((Path(name)/"SUITE_COMPLETE.json").exists())

    def test_synthetic_cpu_pipeline_saves_all_artifacts_and_twenty_frame_predictions(self):
        # Random synthetic features/labels only; never touches clinical records.
        from .train import train_task
        name, target = "a01_regularized_rgb_prior", "hemoglobin_low"
        settings = config.get_variant(name)
        settings.update(epochs=2, patience=2, warmup=0)
        records = pd.DataFrame({"hospital_id": [str(i) for i in range(8)], "video_id": [str(i) for i in range(8)],
            "split": ["train"]*4+["val"]*2+["test"]*2, "clinical_event_id": [f"event{i}" for i in range(8)],
            "raw_value": [80., 90., 100., 110., 85., 105., 88., 108.], "score_threshold": 95., "match_delta_h": 1.})
        index = SimpleNamespace(frame_range=lambda video: (int(video)*20, (int(video)+1)*20))
        with tempfile.TemporaryDirectory() as name_dir:
            root = Path(name_dir)
            cache, source, output = root/"cache", root/"base", root/"output"
            cache.mkdir()
            (source/"source_records").mkdir(parents=True)
            records.to_csv(source/f"source_records/{target}.csv", index=False)
            rng = np.random.default_rng(77)
            for branch in config.BRANCHES:
                values = rng.normal(size=(160, 5, config.reference.DIMENSIONS[branch])).astype(np.float32)
                np.save(cache/f"{branch}.npy", values)
            with patch.object(config.reference, "CACHE", cache), patch.object(config, "BASELINE", source), patch.object(
                    config, "OUTPUT", output), patch.object(config, "get_variant", return_value=settings):
                train_task(name, target, index, None, torch.device("cpu"), {"job_seeds": {target: 123}, "contract": "synthetic"})
            run = output/name/"runs"/target
            for field in ("model.pt", "stage_frozen_best.pt", "metrics.csv", "history.csv", "video_predictions.csv", "job_complete.json"):
                self.assertTrue((run/field).is_file(), field)
            predictions = pd.read_csv(run/"video_predictions.csv")
            self.assertEqual(len(predictions), 8)
            self.assertTrue(predictions.frame_count.eq(20).all())
            history = pd.read_csv(run/"history.csv")
            self.assertTrue(history.train_model_inputs.eq(400).all())
            self.assertTrue(history.optimizer_steps.gt(0).all())
            state = torch.load(run/"model.pt", weights_only=True)
            self.assertEqual(state["source_contract"], "synthetic")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
