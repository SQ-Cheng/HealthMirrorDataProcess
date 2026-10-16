"""Soft weighting and hybrid frame/video supervision, including IR adaptation."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from study.exp2_sota_ablation.data import train_weights as original_weights


def train_weights(records, settings):
    if not any(key in settings for key in ("balance_power", "time_floor", "density_power")):
        return original_weights(records, settings)
    base = {**settings, "balance": False, "density": False, "time_tau": None}
    weights, policy = original_weights(records, base)
    weights = weights.astype(float)
    if settings["balance"]:
        if records.groupby("clinical_event_id").hospital_id.nunique().gt(1).any():
            raise ValueError("Clinical event crosses patients")
        counts = records.groupby("hospital_id").clinical_event_id.transform("nunique") * records.groupby("clinical_event_id").video_id.transform("size")
        power = settings.get("balance_power", 1.)
        weights *= counts.to_numpy(float)**(-power)
        policy.update(patient_event_balance_power=power)
    if settings["density"]:
        low, high = np.quantile(records.raw_value, [.01, .99])
        if high <= low:
            raise ValueError("Density weights require variable training labels")
        edges = np.linspace(low, high, 6)
        bins = np.searchsorted(edges[1:-1], records.raw_value)
        counts = np.bincount(bins, minlength=5)
        factor = (len(records)/counts[bins])**settings.get("density_power", .5)
        factor = np.minimum(factor/factor.mean(), settings.get("density_cap", 3.))
        weights *= factor
        policy.update(density_edges=edges.tolist(), density_counts=counts.tolist(),
                      density_power=settings.get("density_power", .5), density_cap=settings.get("density_cap", 3.))
    if settings["time_tau"] is not None:
        delta = records.match_delta_h.to_numpy(float)
        if not np.isfinite(delta).all() or np.any(delta < 0):
            raise ValueError("Invalid training match distances")
        floor = settings.get("time_floor", .25)
        weights *= floor+(1-floor)*np.exp(-delta/settings["time_tau"])
        policy.update(time_floor=floor, time_tau=settings["time_tau"])
    weights /= weights.mean()
    policy.update(min_weight=float(weights.min()), max_weight=float(weights.max()),
                  effective_video_n=float(weights.sum()**2/np.square(weights).sum()))
    return weights.astype(np.float32), policy


def group_losses(prediction, labels, settings):
    if len(labels) % 20 or not labels.reshape(-1, 20).eq(labels.reshape(-1, 20)[:, :1]).all():
        raise ValueError("Incomplete/mixed-label video group")
    predictions, truth = prediction.float().reshape(-1, 20), labels.reshape(-1, 20)[:, 0]
    frame = F.smooth_l1_loss(predictions, truth[:, None].expand_as(predictions), beta=.5, reduction="none").mean(1)
    video = F.smooth_l1_loss(predictions.mean(1), truth, beta=.5, reduction="none")
    if settings["loss"] == "frame":
        return frame
    if settings["loss"] == "hybrid":
        alpha = settings["frame_fraction"]
        return alpha*frame+(1-alpha)*video
    return video


def train_epoch(model, batches, optimizer, amp, weights, settings, device, online=False):
    model.train()
    count = steps = batch_count = 0
    total = 0.
    for features, labels, rows, views, pixels in batches:
        groups = rows.reshape(-1, 20)
        if not groups.eq(groups[:, :1]).all():
            raise ValueError("Mixed video rows")
        group_weights = weights[groups[:, 0]].to(device)
        optimizer.zero_grad(set_to_none=True)
        if online:
            if settings["rank"]:
                raise ValueError("Online memory-bounded phase does not support cross-video ranking")
            objective = 0.
            for start in range(0, len(labels), 20):
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    prediction = model.forward_video(features[start:start+20], pixels[start:start+20], views[start:start+20], device)
                    loss = group_losses(prediction, labels[start:start+20].to(device), settings)[0]
                    loss = loss*group_weights[start//20]/group_weights.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite online objective")
                amp.scale(loss).backward()
                objective += float(loss.detach())
        else:
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(features.to(device))
                truth = labels.to(device)
                loss = (group_losses(prediction, truth, settings)*group_weights).sum()/group_weights.sum()
                if settings["rank"]:
                    y, pred = truth.reshape(-1, 20)[:, 0], prediction.float().reshape(-1, 20).mean(1)
                    delta = y[:, None]-y[None, :]
                    mask = torch.triu(torch.ones_like(delta, dtype=torch.bool), diagonal=1) & delta.abs().ge(.25)
                    if mask.any():
                        difference = pred[:, None]-pred[None, :]
                        loss += settings["rank"]*F.softplus(-delta.sign()[mask]*difference[mask]/.2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite cached-feature objective")
            amp.scale(loss).backward()
            objective = float(loss.detach())
        amp.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        before = amp.get_scale()
        amp.step(optimizer)
        amp.update()
        steps += int(amp.get_scale() >= before)
        count += len(labels)
        total += objective
        batch_count += 1
    if not steps or count != len(batches.dataset):
        raise RuntimeError("No optimizer updates or dropped/duplicated training frames")
    return total/max(batch_count, 1), count, steps
