"""Memory-bounded paired inputs with one optimizer update per 240-frame batch."""

from contextlib import contextmanager
import time

import numpy as np
import torch
from torch import nn

from study.exp2_face_pretrained_head32_regression import train as primary
from study.exp2_face_rgb_ir_head32_regression.data import RGBIRDataset
from .config import MICROBATCH
from .models import build_model


RGB_PREPARE = primary._prepare_images


def execution_model(model, architecture, target, stage, device):
    if device.type != "cuda":
        return model, "eager_cpu_test"
    # Accumulated backwards must not retain per-shape CUDA Graph memory pools.
    torch._dynamo.config.suppress_errors = False
    torch._inductor.config.compile_threads = 1
    compiled = torch.compile(model, backend="inductor", dynamic=True,
                             options={"triton.cudagraphs": False})
    print(f"[compile-enabled] {target}/{stage} backend=inductor dynamic=True "
          "cudagraphs=False fallback=False evaluation=eager", flush=True)
    return compiled, "inductor:dynamic_no_cudagraphs"


def prepare_images(images, view_codes, interpolation, device):
    if images.ndim != 4 or images.shape[1] != 4 or view_codes.ndim != 1:
        raise ValueError("Expected paired RGB3+IR1 frames and one synchronized view per pair")
    rgb = RGB_PREPARE(images[:, :3], view_codes, interpolation, device)
    ir = RGB_PREPARE(images[:, 3:4].expand(-1, 3, -1, -1), view_codes, interpolation, device)
    return torch.cat((rgb, ir), dim=1).contiguous(memory_format=torch.channels_last)


def train_epoch(model, head, loader, optimizer, scaler, criterion, device, encoder_frozen,
                max_batches=None, frozen_modules=(), freeze_batchnorm_stats=False, train_video_weights=None):
    if train_video_weights is not None or getattr(criterion, "video_level", False):
        raise ValueError("This experiment preserves unweighted frame-level loss")
    if device.type == "cuda" and not hasattr(model, "_orig_mod"):
        raise RuntimeError("CUDA training requires the compiled model; eager fallback is forbidden")
    if encoder_frozen:
        model.eval()
        head.train()
    else:
        model.train()
        for module in frozen_modules:
            module.eval()
        if freeze_batchnorm_stats:
            primary._freeze_batchnorm_running_stats(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss_total = batches = steps = inputs_count = 0
    for batch_index, (images, labels, indices, views) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        batch_sum = 0.
        count = len(labels)
        for offset in range(0, count, MICROBATCH):
            end = min(offset+MICROBATCH, count)
            inputs = prepare_images(images[offset:end], views[offset:end], loader.dataset.interpolation, device)
            truth = labels[offset:end].to(device, non_blocking=True).unsqueeze(1)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses = criterion(model(inputs), truth)
                loss = losses.sum() / count
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; no samples silently skipped")
            scaler.scale(loss).backward()
            if batch_index == 0 and offset == 0 and device.type == "cuda":
                graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
                if not graphs:
                    raise RuntimeError("No compiled graph executed")
                print(f"[compile-confirmed] stage={'head' if encoder_frozen else 'finetune'} "
                      f"microbatch={end-offset} forward/backward=ok unique_graphs={graphs}", flush=True)
            batch_sum += float(losses.detach().sum().cpu())
            inputs_count += end-offset
            del inputs, truth, losses, loss
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), primary.GRAD_CLIP_NORM)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        steps += int(scaler.get_scale() >= before)
        loss_total += batch_sum/count
        batches += 1
    if max_batches is None and inputs_count != loader.dataset.model_input_count:
        raise RuntimeError("Training omitted or duplicated frame/views")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device)/1024**3 if device.type == "cuda" else 0.
    return loss_total/max(batches, 1), steps, inputs_count, elapsed, peak


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model = getattr(model, "_orig_mod", model)
    model.eval()
    scores, truths, rows = [], [], []
    total = count = 0
    for batch_index, (images, labels, indices, views) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for offset in range(0, len(labels), MICROBATCH):
            end = min(offset+MICROBATCH, len(labels))
            prepare = RGB_PREPARE if images.shape[1] == 3 else prepare_images
            inputs = prepare(images[offset:end], views[offset:end], loader.dataset.interpolation, device)
            truth = labels[offset:end].to(device, non_blocking=True).unsqueeze(1)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(inputs)
                losses = criterion(prediction, truth)
            if not torch.isfinite(prediction).all() or not torch.isfinite(losses).all():
                raise RuntimeError("Nonfinite evaluation")
            total += float(losses.sum().cpu())
            count += losses.numel()
            scores.append(prediction.float().cpu().numpy().ravel())
            truths.append(labels[offset:end].numpy().copy())
            rows.append(indices[offset:end].numpy().copy())
    return {"loss": total/count if count else np.nan,
            "scores": np.concatenate(scores), "labels": np.concatenate(truths), "record_indices": np.concatenate(rows)}


@contextmanager
def adapters(ir_index):
    names = ("AllFramesDataset", "_prepare_images", "build_pretrained_model", "_train_epoch", "_evaluate", "_execution_model", "TORCH_COMPILE_ENABLED")
    original = {name: getattr(primary, name) for name in names}
    primary.AllFramesDataset = lambda index, records, **kwargs: RGBIRDataset(index, records, ir_index, **kwargs)
    primary._prepare_images = prepare_images
    primary.build_pretrained_model = build_model
    primary._train_epoch = train_epoch
    primary._evaluate = evaluate
    primary._execution_model = execution_model
    primary.TORCH_COMPILE_ENABLED = True
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(primary, name, value)
