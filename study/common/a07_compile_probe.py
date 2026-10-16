"""Bounded synthetic compile diagnostics, no patient data or optimizer steps."""

import argparse
import json
import time

import torch

from study.exp2_face_dinov3_frozen.backbone import load_encoder
from study.exp2_face_dinov3_farl_regression.backbone import load_farl
from study.common.ir_compile import prepare_farl


def joint_probe(args):
    from study.exp2_sota_5fold_ablation import config
    from study.exp2_sota_5fold_ablation.online import OnlineIR
    from study.exp2_sota_ablation.models import Fusion
    from study.exp2_sota_5fold_ablation.objectives import group_losses
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Joint real-encoder smoke requires a CUDA device")
    settings = config.settings("c08_ir_tail")
    model = OnlineIR(Fusion(settings).to(device), settings, device)
    pixels = torch.randint(0, 256, (20, 1, 224, 224), dtype=torch.uint8).repeat(1, 3, 1, 1)
    features = torch.randn(20, 2560)
    for step in range(args.steps):
        model.train()
        model.zero_grad(set_to_none=True)
        code = step % 3
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = model.forward_video(features, pixels, torch.full((20,), code), device)
        loss = group_losses(prediction, torch.zeros(20, device=device), settings).mean()
        loss.backward()
        torch.cuda.synchronize(device)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not all(g is not None and bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("Missing/nonfinite joint gradient")
        if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
            raise RuntimeError("Frozen prefix received gradients")
        print(json.dumps({"joint_train": step+1, "view": code, "images": 20,
            "seconds": time.perf_counter()-started, "unique_graphs": int(torch._dynamo.utils.counters['stats']['unique_graphs']),
            "finite_output": bool(torch.isfinite(prediction).all()), "finite_gradients": True,
            "optimizer_updates": 0, "peak_gpu_mib": torch.cuda.max_memory_allocated(device)/1024**2}), flush=True)
    model.eval()
    with torch.no_grad():
        for step in range(2):
            started = time.perf_counter()
            prediction = model.forward_video(features, pixels, torch.zeros(20, dtype=torch.long), device)
            torch.cuda.synchronize(device)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Nonfinite joint evaluation")
            print(json.dumps({"joint_eval": step+1, "seconds": time.perf_counter()-started,
                "unique_graphs": int(torch._dynamo.utils.counters['stats']['unique_graphs'])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("dino", "farl", "joint"), default="dino")
    parser.add_argument("--dynamic", choices=("true", "false"), default="false")
    parser.add_argument("--backend", choices=("count", "inductor", "eager"), default="count")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--pure-farl", action="store_true", help="Diagnostic-only equivalent attention without forward attribute mutation")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--microbatches", type=int, default=1, help="Use5 to retain all20 frame graphs before backward")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    torch._logging.set_logs(recompiles=True)
    torch._dynamo.config.suppress_errors = False
    torch._inductor.config.compile_threads = 1
    if args.encoder == "joint":
        joint_probe(args)
        return
    device = torch.device(args.device)
    encoder = (load_encoder(architecture="dinov3_vitb16") if args.encoder == "dino" else load_farl()[0]).to(device)
    inputs = torch.randn(4, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        expected = encoder(inputs).detach()
    if args.pure_farl:
        if args.encoder != "farl":
            raise ValueError("pure-farl applies only to the CLIP/FaRL encoder")
        prepare_farl(encoder)
    if args.encoder == "dino":
        encoder.blocks[-1].requires_grad_(True)
        encoder.norm.requires_grad_(True)
    else:
        encoder.transformer.resblocks[-1].requires_grad_(True)
        encoder.ln_post.requires_grad_(True)
        encoder.proj.requires_grad_(True)
    count = 0

    def counting(graph, examples):
        nonlocal count
        count += 1
        print(json.dumps({"backend_graph": count, "nodes": len(list(graph.graph.nodes))}), flush=True)
        return graph.forward

    model = encoder
    if args.backend != "eager":
        kwargs = {"options": {"triton.cudagraphs": False}} if args.backend == "inductor" else {}
        model = torch.compile(encoder, dynamic=args.dynamic == "true",
            backend=counting if args.backend == "count" else "inductor", fullgraph=args.fullgraph, **kwargs)
    print(json.dumps({"encoder": args.encoder, "backend": args.backend,
        "dynamic": args.dynamic, "device": str(device), "torch": torch.__version__,
        "shape": list(inputs.shape), "stride": list(inputs.stride()), "optimizer_updates": 0,
        "pure_farl": args.pure_farl, "fullgraph": args.fullgraph}), flush=True)
    for step in range(args.steps):
        encoder.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = torch.cat([model(inputs) for _ in range(args.microbatches)], dim=0)
            loss = output.float().square().mean()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        grads = [p.grad for p in encoder.parameters() if p.requires_grad]
        print(json.dumps({"step": step+1, "seconds": time.perf_counter()-started,
            "graphs": count if args.backend == "count" else int(torch._dynamo.utils.counters['stats']['unique_graphs']),
            "images": len(output), "finite_output": bool(torch.isfinite(output).all()),
            "max_abs_difference_from_original": float((output.float()-expected.repeat(args.microbatches, 1).float()).abs().max()),
            "finite_gradients": all(g is not None and bool(torch.isfinite(g).all()) for g in grads)}), flush=True)


if __name__ == "__main__":
    main()
