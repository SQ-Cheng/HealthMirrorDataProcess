"""CPU checks only; randomized official weights are never used for real training."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from study.common.video_loss import VideoBCELoss
from . import config
from .backbone import Head, FrozenPredictor, load_encoder, weight_sha256


class FrozenDinoTests(unittest.TestCase):
    def test_head_size_and_gradient(self):
        head = Head()
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 12417)
        loss = VideoBCELoss(.2)(head(torch.randn(20, 384)).squeeze(1), torch.ones(20)).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_formal_weight_guard(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "missing.pth"
            with patch.object(config, "WEIGHTS", path):
                with self.assertRaises(FileNotFoundError):
                    weight_sha256()
                path.write_bytes(b"not an official checkpoint")
                with self.assertRaisesRegex(RuntimeError, "hash prefix"):
                    weight_sha256()

    def test_official_architecture_encoder_stays_frozen(self):
        torch.set_num_threads(1)
        encoder = load_encoder(pretrained=False)
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 21601152)
        predictor = FrozenPredictor(encoder).train()
        self.assertFalse(predictor.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in predictor.encoder.parameters()))
        before = {key: value.clone() for key, value in predictor.encoder.state_dict().items()}
        optimizer = torch.optim.AdamW(predictor.head.parameters(), lr=2e-4)
        loss = VideoBCELoss(.2)(predictor(torch.randn(20, 3, 32, 32)).squeeze(1), torch.ones(20)).mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(p.grad is None for p in predictor.encoder.parameters()))
        for key, value in predictor.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with torch.no_grad():
            features = predictor.encoder(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(features.shape), (1, 384))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
