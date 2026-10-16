"""CPU/synthetic tests; never launch real training or screen."""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config
from .splits import prepare, choose_inner, validate_records
from .objectives import group_losses, train_weights


def clinical_records(patients=50):
    rows = []
    for patient in range(patients):
        for video in range(2):
            rows.append(dict(hospital_id=f"{patient:010d}", video_id=f"v{patient}_{video}",
                clinical_event_id=f"e{patient}_{video}", raw_value=90+patient+video*3,
                score_threshold=120., match_delta_h=patient % 24,
                split=("train", "val", "test")[patient % 3]))
    return pd.DataFrame(rows)


class SyntheticDataset(Dataset):
    def __init__(self, records, scaler, settings, train):
        from study.exp2_sota_ablation.data import transform
        self.video_records = records.reset_index(drop=True)
        self.labels = transform(records.raw_value, scaler)
        self.views = settings["views"] if train else ["original"]

    def __len__(self):
        return len(self.video_records)*20*len(self.views)

    def __getitem__(self, position):
        frame, view = divmod(position, len(self.views))
        video = frame//20
        features = torch.full((2560,), .01*(video+1))
        return features, torch.tensor(self.labels[video]), torch.tensor(video), torch.tensor(view), torch.empty(0)

    def close(self):
        pass


def synthetic_loader(index, records, settings, scaler, train=False, online=False, ir_index=None):
    dataset = SyntheticDataset(records, scaler, settings, train)
    views = len(dataset.views)
    batches = [[(video*20+frame)*views+view for frame in range(20)]
               for video in range(len(records)) for view in range(views)]
    return dataset, DataLoader(dataset, batch_sampler=batches)


class ProtocolTests(unittest.TestCase):
    def test_balanced_folds_inner_holdout_refit_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "TARGETS", ("hemoglobin_low",)):
            root = Path(directory)/"splits"
            records = clinical_records()
            result = prepare(lambda _: records, root, candidates=3)
            self.assertEqual(result, prepare(lambda _: records, root, candidates=3))
            counts = []
            held = []
            for fold in range(5):
                options = dict(dtype={"hospital_id": str, "video_id": str})
                outer = pd.read_csv(root/f"hemoglobin_low_fold{fold}_refit.csv", **options)
                inner = pd.read_csv(root/f"hemoglobin_low_fold{fold}_selection.csv", **options)
                train = set(outer.loc[outer.split.eq("train"), "hospital_id"])
                test = set(outer.loc[outer.split.eq("test"), "hospital_id"])
                validate_records(inner, train)
                self.assertFalse(test & set(inner.hospital_id))
                self.assertEqual(inner.split.value_counts().to_dict(), {"train": 64, "val": 16})
                self.assertEqual(len(train), 40)
                counts.append(len(test))
                held.extend(test)
            self.assertEqual(counts, [10]*5)
            self.assertEqual(len(set(held)), len(held))
            self.assertEqual(len(held), 50)
            victim = root/"hemoglobin_low_fold0_selection.csv"
            victim.write_text(victim.read_text()+"\n")
            with self.assertRaisesRegex(RuntimeError, "changed fold"):
                prepare(lambda _: records, root, candidates=3)

    def test_inner_search_uses_only_outer_train(self):
        data = clinical_records()
        train = data.loc[data.hospital_id.astype(int) < 40]
        first = choose_inner(train, "hemoglobin_low", 0, candidates=3)
        data.loc[data.hospital_id.astype(int) >= 40, "raw_value"] = 1e15
        second = choose_inner(data.loc[data.hospital_id.astype(int) < 40], "hemoglobin_low", 0, candidates=3)
        self.assertEqual(first, second)

    def test_cumulative_prefix_and_bundles(self):
        self.assertEqual(len(config.VARIANTS), 12)
        previous = "b00_reference"
        for name in list(config.VARIANTS)[1:9]:
            self.assertEqual(config.PARENTS[name], previous)
            self.assertNotEqual(config.settings(previous), config.settings(name))
            previous = name
        for settings in config.VARIANTS.values():
            self.assertFalse(settings["online_epochs"] and settings["rank"])

    def test_hybrid_loss_and_soft_train_only_weights(self):
        prediction = torch.tensor([-1., 1.]*10, requires_grad=True)
        labels = torch.zeros(20)
        frame = group_losses(prediction, labels, {"loss": "frame"})
        hybrid = group_losses(prediction, labels, {"loss": "hybrid", "frame_fraction": .5})
        torch.testing.assert_close(hybrid, frame*.5)
        hybrid.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        records = clinical_records(4).assign(split="train")
        weights, policy = train_weights(records, config.settings("c07_soft_density"))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)
        self.assertEqual(policy["time_floor"], .5)
        records.loc[0, "split"] = "test"
        with self.assertRaises(ValueError):
            train_weights(records, config.settings("c07_soft_density"))

    def test_normalized_warmstart_preserves_rgb_expert_prediction(self):
        from study.exp2_face_dinov3_farl_regression.models import ScalarGatedHead
        from .train import build
        torch.manual_seed(7)
        original = ScalarGatedHead(dino_features=768).eval()
        scaler = {"unit": "g/L", "median": 100., "iqr": 20.}
        def stats(model, index, train):
            model.feature_mean.copy_(torch.linspace(-.2, .2, 2560))
            model.feature_std.copy_(torch.linspace(.5, 1.5, 2560))
        with tempfile.TemporaryDirectory() as directory, patch(
                "study.exp2_sota_5fold_ablation.train.fit_feature_stats", side_effect=stats):
            path = Path(directory)/"rgb.pt"
            torch.save({"state_dict": original.state_dict(), "target_scaler": scaler}, path)
            model = build(config.settings("c05_fold_rgb_warmstart"), "hemoglobin_low", None,
                clinical_records(4).assign(split="train"), scaler, torch.device("cpu"), path).eval()
            features = torch.randn(8, 1280)
            normalized = (features-model.feature_mean[:1280])/model.feature_std[:1280]
            with torch.no_grad():
                expected = original(features).flatten()
                actual = model.head(model.rgb_expert(normalized)).flatten()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_default_does_not_prepare_or_train(self):
        from .run import main
        output = io.StringIO()
        with patch("sys.argv", ["run"]), patch(
                "study.exp2_sota_5fold_ablation.run.validate_sources") as validate, contextlib.redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertFalse(result["training_started"])
        self.assertEqual(result["final_models"], 600)
        validate.assert_not_called()

    def test_queue_crosses_variant_boundary_without_waiting(self):
        from .run import run_queue
        events = []
        jobs = [{"variant": "a", "target": "slow"}, {"variant": "a", "target": "fast"},
                {"variant": "b", "target": "next"}]
        def worker(job):
            events.append((job["target"], "start"))
            time.sleep(.12 if job["target"] == "slow" else .01)
            events.append((job["target"], "end"))
        with patch("study.exp2_sota_5fold_ablation.run.worker", side_effect=worker), ThreadPoolExecutor(2) as executor:
            run_queue(jobs, executor, lambda job: None, 2)
        self.assertLess(events.index(("next", "start")), events.index(("slow", "end")))

    def test_failure_stops_submitting(self):
        from .run import run_queue
        with (patch("study.exp2_sota_5fold_ablation.run.worker", side_effect=RuntimeError("failure")) as worker,
                ThreadPoolExecutor(1) as executor):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                run_queue([{}, {}, {}], executor, lambda job: None, 1)
        self.assertEqual(worker.call_count, 1)

    def test_synthetic_full_selection_refit_outer_test(self):
        from .train import train_job, evaluate
        records = clinical_records(6)
        outer = records.assign(split=np.where(records.hospital_id.astype(int) >= 4, "test", "train"))
        inner = outer.loc[outer.split.eq("train")].assign(split=lambda frame: np.where(frame.hospital_id.astype(int).eq(3), "val", "train"))
        settings = {**config.settings("b00_reference"), "epochs": 2, "patience": 1}
        calls = []
        def observed(*args, **kwargs):
            split = args[6]
            calls.append((split, set(args[1].video_records.hospital_id)))
            return evaluate(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "OUTPUT", Path(directory)), patch(
                "study.exp2_sota_5fold_ablation.train.read_fold", side_effect=lambda t, f, p: inner.copy() if p == "selection" else outer.copy()), patch(
                "study.exp2_sota_5fold_ablation.train.loader", side_effect=synthetic_loader), patch(
                "study.exp2_sota_5fold_ablation.train.evaluate", side_effect=observed), patch.object(config, "settings", return_value=settings):
            train_job("b00_reference", "hemoglobin_low", 0, None, None, torch.device("cpu"), "test", "split")
            root = Path(directory)/"b00_reference/hemoglobin_low/fold0"
            checkpoint = torch.load(root/"model.pt", weights_only=True)
            self.assertEqual(len(checkpoint["training_patient_ids"]), 4)
            self.assertEqual(sum(split == "test" for split, _ in calls), 1)
            test_ids = set(outer.loc[outer.split.eq("test"), "hospital_id"])
            for split, ids in calls:
                if split != "test":
                    self.assertFalse(ids & test_ids)
            refit = pd.read_csv(root/"refit/frozen_history.csv")
            self.assertFalse(any(key.startswith(("test_", "val_")) for key in refit))
            self.assertEqual(int(refit.iloc[-1].epoch), checkpoint["budget"]["frozen_epochs"])
            prediction = pd.read_csv(root/"video_predictions.csv")
            self.assertTrue(prediction.frame_count.eq(20).all())
            self.assertTrue((root/"complete.json").exists())
            self.assertIn("explained_variance", pd.read_csv(root/"metrics.csv"))

    def test_incremental_oof_comparison_only_png_and_patient_integrity(self):
        from .plots import summarize, update
        from study.common.run_video_loss_12h import sha256
        records = clinical_records(10)
        names = list(config.VARIANTS)[:2]
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "OUTPUT", Path(directory)), patch.object(
                config, "TARGETS", ("hemoglobin_low",)), patch.object(config, "VARIANTS", {name: config.VARIANTS[name] for name in names}), patch(
                "study.exp2_sota_ablation.data.read_records", return_value=records):
            for name in names:
                for fold in range(5):
                    root = Path(directory)/name/"hemoglobin_low"/f"fold{fold}"
                    root.mkdir(parents=True)
                    frame = records.loc[records.hospital_id.astype(int).mod(5).eq(fold)].copy()
                    frame = frame.assign(split="test", fold=fold, target="hemoglobin_low", variant=name,
                        y_true=frame.raw_value, y_pred=frame.raw_value+(.5 if name == names[0] else .3), frame_count=20)
                    frame.to_csv(root/"video_predictions.csv", index=False)
                    pd.DataFrame([dict(variant=name, target="hemoglobin_low", fold=fold, split="test",
                        **summarize(frame.y_true, frame.y_pred))]).to_csv(root/"metrics.csv", index=False)
                    files = {name: sha256(root/name) for name in ("metrics.csv", "video_predictions.csv")}
                    (root/"complete.json").write_text(json.dumps({"contract": "test", "files": files}))
            update("test")
            update("test")
            summary = pd.read_csv(Path(directory)/"tables/fold_mean_sd.csv")
            self.assertTrue(summary.fold_count.eq(5).all())
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(pd.read_csv(Path(directory)/names[0]/"oof/hemoglobin_low.csv")), len(records))
            self.assertTrue(list(Path(directory).rglob("*.png")))
            self.assertFalse(list(Path(directory).rglob("*.pdf")))
            ci = pd.read_csv(Path(directory)/"tables/paired_oof_patient_bootstrap.csv")
            self.assertLess(ci.delta_mae.iloc[0], 0)

    def test_online_microbatch_keeps_twenty_frame_dropout_group(self):
        from .online import OnlineIR
        from study.exp2_sota_ablation.models import Fusion
        class Encoder(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(width))
            def forward(self, pixels):
                return pixels.float().mean((1, 2, 3))[:, None]*self.weight[None, :]
        settings = config.settings("c08_ir_tail")
        model = OnlineIR.__new__(OnlineIR)
        torch.nn.Module.__init__(model)
        model.settings, model.fusion = settings, Fusion(settings)
        model.dino, model.farl = Encoder(768), Encoder(512)
        object.__setattr__(model, "dino_forward", model.dino)
        object.__setattr__(model, "farl_forward", model.farl)
        features = torch.randn(20, 2560)
        pixels = torch.ones(20, 3, 8, 8, dtype=torch.uint8)
        with patch("study.exp2_sota_5fold_ablation.online._prepare_images", side_effect=lambda p, *a, **k: p.float()):
            output = model.forward_video(features, pixels, torch.zeros(20, dtype=torch.long), torch.device("cpu"))
        self.assertEqual(tuple(output.shape), (20,))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(model.dino.weight.grad).all())
        self.assertTrue(torch.isfinite(model.farl.weight.grad).all())

    def test_fold_rgb_initialization_is_new_train_only_and_reused(self):
        from .train import warmstart
        records = clinical_records(6)
        outer = records.assign(split=np.where(records.hospital_id.astype(int) >= 4, "test", "train"))
        inner = outer.loc[outer.split.eq("train")].assign(split=lambda frame: np.where(frame.hospital_id.astype(int).eq(3), "val", "train"))
        settings = {**config.WARM_SETTINGS, "epochs": 2, "patience": 1}
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "OUTPUT", Path(directory)), patch.object(
                config, "WARM_SETTINGS", settings), patch("study.exp2_sota_5fold_ablation.train.read_fold",
                side_effect=lambda t, f, p: inner.copy() if p == "selection" else outer.copy()), patch(
                "study.exp2_sota_5fold_ablation.train.loader", side_effect=synthetic_loader):
            root = warmstart("hemoglobin_low", 0, None, None, torch.device("cpu"), "split")
            first = torch.load(root/"inner_rgb.pt", weights_only=True)
            second = torch.load(root/"outer_rgb.pt", weights_only=True)
            self.assertEqual(first["target_scaler"]["median"], inner.loc[inner.split.eq("train"), "raw_value"].median())
            self.assertEqual(second["target_scaler"]["median"], outer.loc[outer.split.eq("train"), "raw_value"].median())
            modified = (root/"outer_rgb.pt").stat().st_mtime_ns
            self.assertEqual(root, warmstart("hemoglobin_low", 0, None, None, torch.device("cpu"), "split"))
            self.assertEqual(modified, (root/"outer_rgb.pt").stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
