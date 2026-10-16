"""Static, side-effect-free IR Transformer compilation for PyTorch2.4."""

import types

import torch

POLICY = {"dynamic": False, "fullgraph": True, "cudagraphs": False,
          "farl_attention": "local_mask_no_attribute_assignment", "version": 1}


def pure_farl_attention(self, x):
    mask = self.attn_mask
    if mask is not None:
        mask = mask.to(dtype=x.dtype, device=x.device)
    return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]


def prepare_farl(encoder):
    """Keep all pretrained tensors/names unchanged; remove only forward mutation."""
    for block in encoder.transformer.resblocks:
        object.__setattr__(block, "attention", types.MethodType(pure_farl_attention, block))
    return encoder


def compile_ir(encoder):
    torch._dynamo.config.suppress_errors = False
    torch._inductor.config.compile_threads = 1
    return torch.compile(encoder, dynamic=False, fullgraph=True, backend="inductor",
                         options={"triton.cudagraphs": False})
