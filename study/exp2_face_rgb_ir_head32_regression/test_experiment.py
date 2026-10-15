"""Pretrained preservation, IR gradients, view equality and paired decoding."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model, freeze_encoder, unfreeze_all
from study.exp2_face_pretrained_head32_regression.train import _prepare_images
from .config import WEIGHTS_DIR, REFERENCE
from study.exp2_face_history_head32_regression.scaling import RobustTargetScaler
from .data import RGBIRDataset, prepare_images
from .models import build_model
from .run import adapters
from .joint_ir_head import build_joint_model, joint_optimizer, joint_training, IR_LR, HEAD_LR


class RGBIRTests(unittest.TestCase):
    def test_rgb_weights_and_zero_ir_preserve_initial_rgb_prediction(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        rgb, _, _ = build_pretrained_model("efficientnet_b0", WEIGHTS_DIR)
        torch.manual_seed(12)
        fused, head, _ = build_model("efficientnet_b0", WEIGHTS_DIR)
        self.assertEqual(sum(p.numel() for p in fused.parameters()), 4048925)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 41089)
        torch.testing.assert_close(fused.features[0][0].weight[:, :3], rgb.features[0][0].weight, rtol=0, atol=0)
        self.assertFalse(fused.features[0][0].weight[:, 3:].any())
        for before, after in zip(rgb.classifier.parameters(), fused.classifier.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        rgb.eval(); fused.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(rgb(x), fused(torch.cat((x, torch.randn(2, 1, 32, 32)), dim=1)), rtol=1e-5, atol=1e-6)

    def test_ir_is_frozen_in_head_stage_and_has_gradient_when_unfrozen(self):
        model, head, _ = build_model("efficientnet_b0", WEIGHTS_DIR)
        model.eval()
        x = torch.randn(2, 4, 32, 32)
        freeze_encoder(model, head)
        model(x).sum().backward()
        self.assertIsNone(model.features[0][0].weight.grad)
        self.assertFalse(model.features[0][0].weight[:, 3:].any())
        model.zero_grad(set_to_none=True)
        unfreeze_all(model)
        model(x).sum().backward()
        grad = model.features[0][0].weight.grad[:, 3:]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(grad.abs().sum() > 0)

    def test_rgb_trained_initialization_preserves_raw_prediction_after_scale_conversion(self):
        path = REFERENCE / "runs/efficientnet_b0/hemoglobin_low/model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        original_scaler = RobustTargetScaler(**checkpoint["target_scaler"])
        paired_scaler = replace(original_scaler, median=original_scaler.median+3, iqr=original_scaler.iqr*1.2)
        rgb, _, _ = build_pretrained_model("efficientnet_b0", WEIGHTS_DIR)
        rgb.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fused, _, source = build_model("efficientnet_b0", WEIGHTS_DIR, path, paired_scaler)
        self.assertEqual(source, str(path))
        torch.testing.assert_close(fused.features[0][0].weight[:, :3], rgb.features[0][0].weight, rtol=0, atol=0)
        torch.testing.assert_close(fused.classifier[0].weight, rgb.classifier[0].weight, rtol=0, atol=0)
        self.assertFalse(fused.features[0][0].weight[:, 3:].any())
        rgb.eval(); fused.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected_raw = rgb(x)*original_scaler.iqr+original_scaler.median
            actual_raw = fused(torch.cat((x, torch.randn(2, 1, 32, 32)), dim=1))*paired_scaler.iqr+paired_scaler.median
        torch.testing.assert_close(actual_raw, expected_raw, rtol=1e-5, atol=1e-5)
        with self.assertRaises(ValueError):
            build_model("efficientnet_b0", WEIGHTS_DIR, path, replace(paired_scaler, target="lactate_high"))

    def test_all_views_keep_rgb_transform_exactly_unchanged(self):
        images = torch.randint(0, 256, (2, 4, 224, 224), dtype=torch.uint8)
        codes = torch.arange(5).repeat(2, 1)
        transformed = prepare_images(images, codes, "bicubic", torch.device("cpu"))
        rgb = _prepare_images(images[:, :3], codes, "bicubic", torch.device("cpu"))
        torch.testing.assert_close(transformed[:, :3], rgb, rtol=0, atol=0)
        self.assertEqual(transformed.shape, (10, 4, 224, 224))
        self.assertTrue(torch.isfinite(transformed).all())

    def test_joint_optimizer_updates_ir_and_head_but_freezes_rgb_and_bn_exactly(self):
        from study.exp2_face_pretrained_head32_regression import train
        from torch.utils.data import TensorDataset, DataLoader
        path = REFERENCE / "runs/efficientnet_b0/hemoglobin_low/model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        scaler = RobustTargetScaler(**checkpoint["target_scaler"])
        model, head, _ = build_joint_model("efficientnet_b0", WEIGHTS_DIR, path, scaler)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 41377)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4048925)
        optimizer = joint_optimizer(p for p in model.parameters() if p.requires_grad)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [IR_LR, HEAD_LR])
        dataset = TensorDataset(torch.randint(0, 256, (2, 4, 32, 32), dtype=torch.uint8),
            torch.zeros(2), torch.arange(2), torch.zeros(2, dtype=torch.long))
        dataset.interpolation = "bicubic"
        dataset.frame_video_rows = np.zeros(2, dtype=np.int64)
        loader = DataLoader(dataset, batch_size=2)
        grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
        with adapters(None, compile_enabled=False), joint_training():
            result = train._train_epoch(model, head, loader, optimizer, grad_scaler,
                torch.nn.SmoothL1Loss(beta=.5, reduction="none"), torch.device("cpu"), False)
        self.assertEqual(result[1], 1)
        self.assertTrue(model.features[0][0].ir_weight.abs().sum() > 0)
        self.assertIsNone(model.features[0][0].weight.grad)
        self.assertFalse(model.features.training)
        self.assertTrue(head.training)
        after = model.state_dict()
        for key in before:
            if key.startswith("features.") and key != "features.0.0.ir_weight":
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(before[key], after[key]) for key in before if key.startswith("classifier.")))
        restored, _, _ = build_joint_model("efficientnet_b0", WEIGHTS_DIR, path, scaler)
        restored.load_state_dict(after, strict=True)

    def test_joint_adapters_restore_primary_on_error(self):
        from study.exp2_face_pretrained_head32_regression import train
        names = ("AdamW", "unfreeze_all", "_train_epoch", "DIRECT_LEARNING_RATE", "DIRECT_PATIENCE")
        before = {key: getattr(train, key) for key in names}
        with self.assertRaises(RuntimeError):
            with joint_training():
                self.assertIs(train.AdamW, joint_optimizer)
                raise RuntimeError("test")
        for key, value in before.items():
            self.assertIs(getattr(train, key), value)

    def test_paired_decode_uses_aligned_indices_and_single_grayscale_channel(self):
        index = SimpleNamespace(video_lookup={"v": 0}, frame_range=lambda v: (0, 20),
                                video_paths=np.array(["rgb"]), source_indices=np.arange(20))
        ir = SimpleNamespace(video_paths=np.array(["ir"]))
        records = pd.DataFrame({"video_id": ["v"], "robust_scaled_raw_value": [2.]})
        dataset = RGBIRDataset(index, records, ir)
        rgb_image = torch.full((3, 224, 224), 50, dtype=torch.uint8)
        ir_image = torch.full((3, 224, 224), 150, dtype=torch.uint8)
        dataset._handle = lambda path: path
        with patch("study.exp2_face_rgb_ir_head32_regression.data.decode_indexed_frame", side_effect=[rgb_image, ir_image]) as decoder:
            image, label, row, _ = dataset[3]
            self.assertEqual(image.shape, (4, 224, 224))
            torch.testing.assert_close(image[:3], rgb_image)
            torch.testing.assert_close(image[3], ir_image[0])
            self.assertEqual(decoder.call_args_list[0].args[2], decoder.call_args_list[1].args[2])
            self.assertEqual(label.item(), 2.)
            self.assertEqual(row.item(), 3)
        dataset.close()

    def test_local_adapters_restore_primary_even_on_error(self):
        from study.exp2_face_pretrained_head32_regression import train
        original = train.build_pretrained_model
        with self.assertRaises(RuntimeError):
            with adapters(None, compile_enabled=False):
                self.assertIs(train.build_pretrained_model, build_model)
                raise RuntimeError("test")
        self.assertIs(train.build_pretrained_model, original)


if __name__ == "__main__":
    unittest.main()
