"""Small CPU tests for the actual production compile fix and migration guard."""

from copy import deepcopy
from unittest.mock import patch
import unittest

import torch
from clip.model import VisionTransformer

from study.common.ir_compile import prepare_farl, compile_ir
from study.common.migrate_ir_compile_fix import validate_policy, ADDED


class IRCompileTests(unittest.TestCase):
    def test_pure_farl_preserves_weights_output_and_gradients(self):
        torch.manual_seed(37)
        original = VisionTransformer(32, 16, 64, 2, 4, 32).eval()
        fixed = prepare_farl(deepcopy(original))
        self.assertEqual(set(original.state_dict()), set(fixed.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, fixed.state_dict()[key], rtol=0, atol=0)
        pixels = torch.randn(2, 3, 32, 32)
        first, second = original(pixels), fixed(pixels)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        first.square().mean().backward()
        second.square().mean().backward()
        for old, new in zip(original.parameters(), fixed.parameters()):
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
        self.assertTrue(all(block.attn_mask is None for block in fixed.transformer.resblocks))

    def test_fixed_farl_has_one_graph_across_repeated_calls(self):
        torch._dynamo.reset()
        encoder = prepare_farl(VisionTransformer(32, 16, 64, 2, 4, 32).eval())
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        compiled = torch.compile(encoder, backend=backend, fullgraph=True, dynamic=False)
        for _ in range(3):
            encoder.zero_grad(set_to_none=True)
            compiled(torch.randn(2, 3, 32, 32)).sum().backward()
        self.assertEqual(len(graphs), 1)
        torch._dynamo.reset()

    def test_production_static_fullgraph_flags(self):
        model = torch.nn.Linear(3, 2)
        with patch("torch.compile", return_value="compiled") as compiler:
            self.assertEqual(compile_ir(model), "compiled")
        compiler.assert_called_once_with(model, dynamic=False, fullgraph=True, backend="inductor",
                                         options={"triton.cudagraphs": False})

    def test_migration_rejects_unrelated_changes(self):
        old = {"protocols": {"c08_ir_tail": {"lr": 1e-4}, "b00_reference": {"lr": 1e-4}},
               "split_contract": "unchanged", "code": {"study/exp2_sota_ablation/online.py": "old", "data.py": "same"}}
        new = deepcopy(old)
        new["protocols"]["c08_ir_tail"].update(ADDED)
        new["code"]["study/exp2_sota_ablation/online.py"] = "new"
        validate_policy(old, new)
        for changed in ("split", "baseline", "source"):
            bad = deepcopy(new)
            if changed == "split":
                bad["split_contract"] = "changed"
            elif changed == "baseline":
                bad["protocols"]["b00_reference"]["lr"] = .01
            else:
                bad["code"]["data.py"] = "changed"
            with self.assertRaises(RuntimeError):
                validate_policy(old, bad)


if __name__ == "__main__":
    unittest.main()
