"""Freeze, gradient, gate initialization and branch-normalization checks."""

import unittest
import json
from pathlib import Path
import tempfile
import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.train import _prepare_images, IMAGENET_MEAN, IMAGENET_STD
from . import config
from .backbone import load_farl
from .models import ConcatHead, GatedHead, ScalarGatedHead, FrozenDualPredictor


class DualTests(unittest.TestCase):
    def test_heads_parameters_and_gradients(self):
        torch.set_num_threads(1)
        for model, expected in ((ConcatHead(), 28801), (GatedHead(), 68161), (ScalarGatedHead(), 60034)):
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            prediction = model(torch.randn(240, 896))
            self.assertEqual(prediction.shape, (240, 1))
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        gated = GatedHead()
        torch.testing.assert_close(gated.gate(torch.randn(2, 128)).sigmoid(), torch.full((2, 64), .5))

    def test_scalar_gate_broadcast_is_input_dependent(self):
        model = ScalarGatedHead().eval()
        torch.testing.assert_close(model.gate(torch.randn(3, 128)).sigmoid(), torch.full((3, 1), .5))
        with torch.no_grad():
            model.gate.weight.normal_()
        features = torch.randn(3, 896)
        dino = model.dino_project(features[:, :384])
        farl = model.farl_project(features[:, 384:])
        gate = model.gate(torch.cat((dino, farl), dim=1)).sigmoid()
        self.assertEqual(gate.shape, (3, 1))
        self.assertGreater(gate.std().item(), 0)
        expanded = gate.expand(-1, 64)
        torch.testing.assert_close(expanded, gate.repeat(1, 64))
        torch.testing.assert_close(model(features), model.head(expanded * dino + (1 - expanded) * farl))

    def test_extension_preserves_base_contracts_and_training_settings(self):
        from .run import register_contracts, training_config
        from study.common.run_video_loss_12h import sha256
        base = {"variants": list(config.BASE_VARIANTS), "parameters": {"concat": 28801, "gated": 68161}, "learning_rate": .0002}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "experiment_manifest.json"
            path.write_text(json.dumps(base, indent=2) + "\n")
            before = sha256(path)
            contracts = register_contracts(base, root)
            self.assertEqual(sha256(path), before)
            self.assertEqual(contracts["concat"], before)
            self.assertEqual(contracts["gated"], before)
            self.assertNotEqual(contracts["scalar_gated"], before)
            self.assertEqual(register_contracts(base, root), contracts)
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                register_contracts({**base, "learning_rate": .1}, root)
        cfg = training_config()
        for variant in config.VARIANTS:
            self.assertEqual(cfg.LEARNING_RATES[variant], 2e-4)
            self.assertEqual(cfg.MIN_LEARNING_RATES[variant], 1e-6)
        self.assertEqual((cfg.MAX_EPOCHS, cfg.PATIENCE, cfg.WEIGHT_DECAY, cfg.LOSS_LEVEL), (80, 12, .001, "frame"))

    def test_encoders_remain_frozen(self):
        for variant in ("gated", "scalar_gated"):
            predictor = FrozenDualPredictor(nn.Linear(3, 384), nn.Linear(3, 512), variant).train()
            predictor(torch.randn(4, 3), torch.randn(4, 3)).sum().backward()
            self.assertFalse(predictor.dino.training)
            self.assertFalse(predictor.farl.training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for module in (predictor.dino, predictor.farl) for p in module.parameters()))

    def test_branch_pixel_views_match_before_normalization(self):
        pixels = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        views = torch.arange(5).repeat(2, 1)
        dino = _prepare_images(pixels, views, "bicubic", torch.device("cpu"))
        farl = _prepare_images(pixels, views, "bicubic", torch.device("cpu"), config.CLIP_MEAN, config.CLIP_STD)
        mean = lambda values: torch.tensor(values).reshape(1, 3, 1, 1)
        torch.testing.assert_close(dino * mean(IMAGENET_STD) + mean(IMAGENET_MEAN),
                                   farl * mean(config.CLIP_STD) + mean(config.CLIP_MEAN), rtol=1e-6, atol=2e-7)

    def test_real_farl_checkpoint_strict_load_and_forward(self):
        encoder, excluded = load_farl()
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 86192640)
        self.assertEqual(len(excluded), 17)
        with torch.no_grad():
            output = encoder(torch.zeros(1, 3, 224, 224))
        self.assertEqual(output.shape, (1, 512))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
