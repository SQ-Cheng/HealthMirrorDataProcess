"""Independent weights, grayscale adaptation, and exact accumulation scaling."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression.models import freeze_encoder, unfreeze_all
from . import config, engine
from .models import build_model


class Tests(unittest.TestCase):
    def test_independent_pretraining_and_freeze(self):
        model, head, _ = build_model("efficientnet_b0", config.paired.WEIGHTS_DIR)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8097145)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 82049)
        for rgb, ir in zip(model.rgb_encoder.parameters(), model.ir_encoder.parameters()):
            torch.testing.assert_close(rgb, ir, rtol=0, atol=0)
            self.assertNotEqual(rgb.data_ptr(), ir.data_ptr())
        freeze_encoder(model, head)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 82049)
        unfreeze_all(model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(2, 6, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_synchronous_views_and_gray_imagenet_normalization(self):
        image = torch.randint(0, 256, (5, 4, 224, 224), dtype=torch.uint8)
        # Identical synthetic grayscale in both branches must remain identical.
        image[:, :3] = image[:, 3:4]
        output = engine.prepare_images(image, torch.arange(5), "bicubic", torch.device("cpu"))
        torch.testing.assert_close(output[:, :3], output[:, 3:], rtol=0, atol=0)
        for view in range(5):
            rgb = engine.RGB_PREPARE(image[view:view+1, :3], torch.tensor([view]), "bicubic", torch.device("cpu"))
            torch.testing.assert_close(output[view:view+1, :3], rgb, rtol=0, atol=0)

    def test_partial_microbatch_gradient_and_evaluation(self):
        torch.manual_seed(23)
        model = nn.Linear(6, 1)
        reference = nn.Linear(6, 1)
        reference.load_state_dict(model.state_dict())
        images, labels = torch.randn(13, 6), torch.randn(13)
        batch = (images, labels, torch.arange(13), torch.zeros(13, dtype=torch.long))
        class Loader(list):
            dataset = SimpleNamespace(model_input_count=13, interpolation="bicubic")
        loader = Loader([batch])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        criterion = nn.SmoothL1Loss(beta=.5, reduction="none")
        criterion(reference(images), labels[:, None]).mean().backward()
        nn.utils.clip_grad_norm_(reference.parameters(), engine.primary.GRAD_CLIP_NORM)
        ref_optimizer.step()
        with patch.object(engine, "MICROBATCH", 5), patch.object(engine, "prepare_images", side_effect=lambda x, *args: x):
            result = engine.train_epoch(model, model, loader, optimizer, torch.amp.GradScaler("cuda", enabled=False),
                                        criterion, torch.device("cpu"), False)
            evaluation = engine.evaluate(model, loader, criterion, torch.device("cpu"))
        self.assertEqual((result[1], result[2]), (1, 13))
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)
        self.assertEqual(len(evaluation["scores"]), 13)
        torch.testing.assert_close(torch.from_numpy(evaluation["scores"]), model(images).flatten())

    def test_adapters_restore_primary(self):
        names = ("AllFramesDataset", "_prepare_images", "build_pretrained_model", "_train_epoch", "_evaluate", "_execution_model", "TORCH_COMPILE_ENABLED")
        before = {name: getattr(engine.primary, name) for name in names}
        with engine.adapters(None):
            self.assertIs(engine.primary._train_epoch, engine.train_epoch)
            self.assertTrue(engine.primary.TORCH_COMPILE_ENABLED)
            self.assertIs(engine.primary._execution_model, engine.execution_model)
        for name in names:
            self.assertIs(getattr(engine.primary, name), before[name])

    def test_compile_options(self):
        model = nn.Linear(6, 1)
        with patch.object(torch, "compile", return_value=model) as compile_model:
            with patch.object(torch._dynamo.config, "suppress_errors", False), patch.object(torch._inductor.config, "compile_threads", 1):
                result, backend = engine.execution_model(model, "efficientnet_b0", "test", "finetune", torch.device("cuda:0"))
        self.assertIs(result, model)
        self.assertEqual(backend, "inductor:dynamic_no_cudagraphs")
        compile_model.assert_called_once_with(model, backend="inductor", dynamic=True, options={"triton.cudagraphs": False})

    def test_training_log_supports_triton_gpu_query(self):
        from triton.testing import nvsmi
        from .run import TrainingTee
        stream, log = io.StringIO(), io.StringIO()
        with patch("subprocess.check_output", return_value=b"1530\n"), contextlib.redirect_stdout(TrainingTee(stream, log)):
            self.assertEqual(nvsmi(["clocks.max.sm"]), [1530])
            print("compiled")
        self.assertEqual(stream.getvalue(), "compiled\n")
        self.assertEqual(log.getvalue(), stream.getvalue())


if __name__ == "__main__":
    unittest.main()
