"""CPU checks for frozen EN features, head sizes and cross-experiment mapping."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from study.exp2_face_dinov3_frozen.features import load_frozen_encoder
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from . import run_frozen_en_regression_controls as controls


class FrozenENTests(unittest.TestCase):
    def test_encoder_and_batchnorm_stay_frozen(self):
        torch.set_num_threads(1)
        encoder = load_frozen_encoder("efficientnet_b0")
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        self.assertEqual(sum(p.numel() for p in encoder.parameters()), 4007548)
        buffers = {key: value.clone() for key, value in encoder.named_buffers()}
        with torch.no_grad():
            feature = encoder(torch.randn(2, 3, 224, 224))
        self.assertEqual(feature.shape, (2, 1280))
        for hidden, expected in ((32, 41089), (64, 82177)):
            head = controls.build_head(hidden)
            self.assertEqual(sum(p.numel() for p in head.parameters()), expected)
            head(feature).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        for key, value in encoder.named_buffers():
            torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)

    @staticmethod
    def indexes():
        source = FrameOffsetIndex(np.array(["a", "b"]), np.array(["a.mkv", "b.mkv"]), np.array([0, 20, 40]),
                                  np.arange(40), np.arange(40) + 1, np.tile(np.arange(20), 2), np.array(["ffv1", "ffv1"]))
        cache = FrameOffsetIndex(np.array(["b", "a"]), np.array(["b.mkv", "a.mkv"]), np.array([0, 20, 40]),
                                 np.r_[20:40, 0:20], np.r_[21:41, 1:21], np.tile(np.arange(20), 2), np.array(["ffv1", "ffv1"]))
        return source, cache

    def test_mapping_checks_actual_frames_not_global_positions(self):
        source, cache = self.indexes()
        mapping = controls.frame_mapping(source, cache)
        np.testing.assert_array_equal(mapping, np.r_[20:40, 0:20])
        cache.ends[0] += 1
        with self.assertRaises(AssertionError):
            controls.frame_mapping(source, cache)

    def test_exp2_reads_the_mapped_shared_cache(self):
        source, cache = self.indexes()
        records = pd.DataFrame({"hospital_id": ["1", "2"], "video_id": ["a", "b"],
                                "clinical_event_id": ["e1", "e2"], "binary_label": [0, 1],
                                "robust_scaled_raw_value": [0., 1.]})
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            features = np.repeat(np.arange(40, dtype=np.float32)[:, None, None], 5 * 1280, axis=1).reshape(40, 5, 1280)
            np.save(root / "cls_features.npy", features)
            with patch.object(controls, "CACHE", root), patch.object(controls, "INDEX6", cache), \
                 patch.object(controls, "MAPPING", controls.frame_mapping(source, cache)):
                data, batches = controls.single_loader(source, records, controls.ARCHITECTURE, "regression", False)
                self.assertEqual(data[0][0][0].item(), 20.)
                self.assertEqual(data[20][0][0].item(), 0.)
                self.assertEqual(sum(len(batch[0]) for batch in batches), 40)


if __name__ == "__main__":
    unittest.main()
