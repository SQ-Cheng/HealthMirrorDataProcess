"""Inner selection followed by fixed-budget full-80% refit and one outer test."""

import gc
import fcntl
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import explained_variance_score

from study.common.run_video_loss_12h import sha256
from study.exp2_face_dinov3_farl_regression.models import ScalarGatedHead
from study.exp2_sota_ablation.data import fit_transform, fit_feature_stats, loader
from study.exp2_sota_ablation.models import Fusion
from study.exp2_sota_ablation.train import evaluate, schedule_lr
from . import config
from .objectives import train_epoch, train_weights
from .splits import read_fold, target_seed, validate_records


class RGBHead(ScalarGatedHead):
    def __init__(self):
        super().__init__(dino_features=768)

    def forward(self, features):
        return super().forward(features[:, :1280]).flatten()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def snapshot(model, online=False):
    fusion = model.fusion if online else model
    return {"state_dict": {key: value.detach().cpu().clone() for key, value in fusion.state_dict().items()},
            "ir_tail_state": model.tail_state() if online else None}


def restore(model, state, online=False):
    (model.fusion if online else model).load_state_dict(state["state_dict"], strict=True)
    if online:
        model.load_tail(state["ir_tail_state"])


def optimizer_for(model, settings, online):
    if online:
        encoder = [p for name, p in model.named_parameters() if name.startswith(("dino.", "farl.")) and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": model.fusion.parameters(), "lr": settings["online_head_lr"], "weight_decay": settings["weight_decay"]},
            {"params": encoder, "lr": settings["online_encoder_lr"], "weight_decay": settings["online_encoder_weight_decay"]}])
        return optimizer, settings["online_epochs"], settings["online_patience"], settings["online_warmup"], [
            settings["online_head_lr"], settings["online_encoder_lr"]], [settings["online_head_min_lr"], settings["online_encoder_min_lr"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    return optimizer, settings["epochs"], settings["patience"], settings["warmup"], [settings["lr"]], [settings["min_lr"]]


def fit_phase(model, settings, train, val, scaler, target, index, ir_index, device,
              root, phase, fixed_epochs=None, initial_eligible=False):
    """No outer-test argument exists here. Refit has no validation argument either."""
    if set(train.split) != {"train"} or (val is not None and set(val.split) != {"val"}):
        raise ValueError("Incorrect phase roles")
    if val is not None and set(train.hospital_id) & set(val.hospital_id):
        raise ValueError("Inner validation patient leakage")
    selection = fixed_epochs is None
    if selection != (val is not None):
        raise ValueError("Select with inner validation; refit with fixed epochs and no validation")
    online = phase == "ir_tail"
    optimizer, horizon, patience, warmup, initial, floors = optimizer_for(model, settings, online)
    epochs = horizon if selection else int(fixed_epochs)
    if not 0 <= epochs <= horizon:
        raise ValueError("Invalid selected epoch budget")
    weights, policy = train_weights(train, settings)
    root.mkdir(parents=True, exist_ok=True)
    train.assign(training_weight=weights).to_csv(root/f"{phase}_training_weights.csv", index=False)
    (root/f"{phase}_weight_policy.json").write_text(json.dumps(policy, indent=2)+"\n")
    weights = torch.from_numpy(weights)
    train_data, batches = loader(index, train, settings, scaler, True, online, ir_index)
    train_eval, train_eval_batches = loader(index, train, settings, scaler, False, online, ir_index)
    val_data = val_batches = None
    if val is not None:
        val_data, val_batches = loader(index, val, settings, scaler, False, online, ir_index)
    amp = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
    rows, best, best_epoch, best_score, stale = [], None, 0, float("inf"), 0
    if selection and initial_eligible:
        result, _ = evaluate(model, val_data, val_batches, scaler, target, device, "val", online)
        if not np.isfinite(result["mae"]):
            raise RuntimeError("Nonfinite initial inner validation MAE")
        best_score, best = result["mae"], snapshot(model, online)
        rows.append({"stage": phase, "epoch": 0, "train_model_inputs": 0,
                     **{f"val_{key}": value for key, value in result.items()}})
    if not selection and epochs == 0:
        result, _ = evaluate(model, train_eval, train_eval_batches, scaler, target, device, "train", online)
        rows.append({"stage": phase, "epoch": 0, "train_model_inputs": 0,
                     **{f"train_{key}": value for key, value in result.items()}})
    try:
        for epoch in range(1, epochs+1):
            # Refit repeats the selected prefix of the ORIGINAL LR schedule.
            schedule_lr(optimizer, epoch, horizon, warmup, initial, floors)
            started = time.perf_counter()
            loss, inputs, steps = train_epoch(model, batches, optimizer, amp, weights, settings, device, online)
            train_result, _ = evaluate(model, train_eval, train_eval_batches, scaler, target, device, "train", online)
            row = {"stage": phase, "epoch": epoch, "train_optimization_loss": loss,
                "train_model_inputs": inputs, "optimizer_steps": steps,
                "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter()-started,
                **{f"train_{key}": value for key, value in train_result.items()}}
            if selection:
                val_result, _ = evaluate(model, val_data, val_batches, scaler, target, device, "val", online)
                row.update({f"val_{key}": value for key, value in val_result.items()})
                if not np.isfinite(val_result["mae"]):
                    raise RuntimeError("Nonfinite inner validation MAE")
                if val_result["mae"] < best_score-1e-4:
                    best_score, best_epoch, best, stale = val_result["mae"], epoch, snapshot(model, online), 0
                else:
                    stale += 1
            rows.append(row)
            pd.DataFrame(rows).to_csv(root/f"{phase}_history.csv", index=False)
            print(f"[epoch] {root.name}/{phase} {epoch}/{epochs} select={selection} "
                  f"train_mae={train_result['mae']:.5g} inner_val_mae={row.get('val_mae', float('nan')):.5g}", flush=True)
            if selection and stale >= patience:
                break
        if selection:
            if best is None:
                raise RuntimeError("No finite inner-selected checkpoint")
            restore(model, best, online)
        else:
            best, best_epoch = snapshot(model, online), epochs
        pd.DataFrame(rows).to_csv(root/f"{phase}_history.csv", index=False)
        torch.save({**best, "target_transform": scaler, "epoch": best_epoch,
                    "selection_scope": "inner_validation" if selection else "fixed_budget_no_validation"}, root/f"{phase}.pt")
        return {"epochs": best_epoch, "val_mae": best_score if selection else None}
    finally:
        train_data.close()
        train_eval.close()
        if val_data is not None:
            val_data.close()
        del batches, train_eval_batches, val_batches
        gc.collect()


def warmstart(target, fold, index, ir_index, device, split_contract):
    """Fresh fold-local supervised RGB expert; never load old patient-trained weights."""
    root = config.OUTPUT / "rgb_initialization" / target / f"fold{fold}"
    root.mkdir(parents=True, exist_ok=True)
    # Different variants may reach the same fold concurrently in the global queue.
    with (root/".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _warmstart(target, fold, index, ir_index, device, split_contract, root)


def _warmstart(target, fold, index, ir_index, device, split_contract, root):
    marker = root/"complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved["split_contract"] == split_contract and all(
                sha256(root/name) == digest for name, digest in saved["files"].items())):
            return root
        raise RuntimeError("Fold RGB initialization contract changed")
    inner = read_fold(target, fold, "selection")
    outer = read_fold(target, fold, "refit")
    settings = dict(config.WARM_SETTINGS)
    seed = target_seed(target, fold)
    selection_scaler = fit_transform(inner, target, settings)
    seed_all(seed)
    model = RGBHead().to(device)
    selected = fit_phase(model, settings, inner.loc[inner.split.eq("train")].reset_index(drop=True),
        inner.loc[inner.split.eq("val")].reset_index(drop=True), selection_scaler, target,
        index, ir_index, device, root/"selection", "frozen")
    selected_checkpoint = torch.load(root/"selection/frozen.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": selected_checkpoint["state_dict"], "target_scaler": selection_scaler}, root/"inner_rgb.pt")
    train = outer.loc[outer.split.eq("train")].reset_index(drop=True)
    scaler = fit_transform(train, target, settings)
    seed_all(seed)
    model = RGBHead().to(device)
    fit_phase(model, settings, train, None, scaler, target, index, ir_index, device,
        root/"refit", "frozen", fixed_epochs=selected["epochs"])
    torch.save({"state_dict": model.cpu().state_dict(), "target_scaler": scaler}, root/"outer_rgb.pt")
    (root/"selected_budget.json").write_text(json.dumps(selected, indent=2)+"\n")
    files = {name: sha256(root/name) for name in ("inner_rgb.pt", "outer_rgb.pt", "selected_budget.json")}
    marker.write_text(json.dumps({"split_contract": split_contract, "files": files,
        "heldout_patients_used": False, "extra_original_rgb_training_videos": False}, indent=2)+"\n")
    return root


def build(settings, target, index, train, scaler, device, warm=None):
    # Import raw RGB projections first, then change coordinates exactly to z-score inputs.
    model = Fusion({**settings, "normalize": False} if warm is not None else settings)
    if settings["normalize"]:
        fit_feature_stats(model, index, train)
    if warm is not None:
        model.warm_start_rgb(torch.load(warm, map_location="cpu", weights_only=True), scaler)
        if settings["normalize"]:
            with torch.no_grad():
                for expert, start in ((model.rgb_expert, 0), (model.ir_expert, 1280)):
                    for layer, offset, width in ((expert.dino_project[0], start, 768),
                                                 (expert.farl_project[0], start+768, 512)):
                        original = layer.weight.detach().clone()
                        layer.bias.add_(original @ model.feature_mean[offset:offset+width])
                        layer.weight.mul_(model.feature_std[offset:offset+width])
        model.settings = settings
    return model.to(device)


def train_job(name, target, fold, index, ir_index, device, contract, split_contract):
    root = config.OUTPUT / name / target / f"fold{fold}"
    root.mkdir(parents=True, exist_ok=True)
    settings = config.settings(name)
    inner, outer = read_fold(target, fold, "selection"), read_fold(target, fold, "refit")
    validate_records(inner, outer.loc[outer.split.eq("train"), "hospital_id"].unique())
    validate_records(outer)
    warm = warmstart(target, fold, index, ir_index, device, split_contract) if settings["warm_start"] else None
    train, val = [inner.loc[inner.split.eq(split)].reset_index(drop=True) for split in ("train", "val")]
    scaler = fit_transform(inner, target, settings)
    seed_all(target_seed(target, fold))
    model = build(settings, target, index, train, scaler, device, warm/"inner_rgb.pt" if warm else None)
    selected = fit_phase(model, settings, train, val, scaler, target, index, ir_index, device,
        root/"selection", "frozen", initial_eligible=settings["warm_start"])
    plan = {"frozen_epochs": selected["epochs"], "ir_tail_epochs": 0,
        "selected_stage": "frozen", "inner_val_mae": selected["val_mae"]}
    if settings["online_epochs"]:
        from .online import OnlineIR
        model = OnlineIR(model, settings, device)
        tail = fit_phase(model, settings, train, val, scaler, target, index, ir_index, device,
            root/"selection", "ir_tail", initial_eligible=True)
        if tail["epochs"] > 0 and tail["val_mae"] < selected["val_mae"]-1e-4:
            plan.update(ir_tail_epochs=tail["epochs"], selected_stage="ir_tail", inner_val_mae=tail["val_mae"])
    (root/"selected_budget.json").write_text(json.dumps(plan, indent=2)+"\n")
    del model
    gc.collect()
    if device.type == "cuda":
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    # Reinitialize; no selection-stage optimizer/model is reused on the outer 80%.
    train = outer.loc[outer.split.eq("train")].reset_index(drop=True)
    scaler = fit_transform(train, target, settings)
    seed_all(target_seed(target, fold))
    model = build(settings, target, index, train, scaler, device, warm/"outer_rgb.pt" if warm else None)
    fit_phase(model, settings, train, None, scaler, target, index, ir_index, device,
        root/"refit", "frozen", fixed_epochs=plan["frozen_epochs"])
    online = plan["selected_stage"] == "ir_tail"
    if online:
        from .online import OnlineIR
        model = OnlineIR(model, settings, device)
        fit_phase(model, settings, train, None, scaler, target, index, ir_index, device,
            root/"refit", "ir_tail", fixed_epochs=plan["ir_tail_epochs"])
    # Persist the FINAL fixed-budget model before opening the test evaluator.
    torch.save({**snapshot(model, online), "settings": settings, "target_transform": scaler,
        "variant": name, "target": target, "fold": fold, "budget": plan,
        "seed": target_seed(target, fold), "contract": contract, "split_contract": split_contract,
        "training_patient_ids": sorted(train.hospital_id.unique().tolist()),
        "selection": "inner MAE only; final full-80% refit checkpoint, no test selection"}, root/"model.pt")
    metrics, predictions = [], []
    for split in ("train", "test"):
        group = outer.loc[outer.split.eq(split)].reset_index(drop=True)
        dataset, batches = loader(index, group, settings, scaler, False, online, ir_index)
        try:
            result, prediction = evaluate(model, dataset, batches, scaler, target, device, split, online)
            result["explained_variance"] = float(explained_variance_score(prediction.y_true, prediction.y_pred))
            metrics.append({"target": target, "fold": fold, "variant": name, "split": split, **result})
            predictions.append(prediction.assign(target=target, fold=fold, variant=name))
        finally:
            dataset.close()
            del batches
    pd.DataFrame(metrics).to_csv(root/"metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(root/"video_predictions.csv", index=False)
    files = {name: sha256(root/name) for name in ("model.pt", "selected_budget.json", "metrics.csv", "video_predictions.csv")}
    (root/"complete.json").write_text(json.dumps({"contract": contract, "files": files})+"\n")
    print(f"[fold-complete] {name}/{target}/fold{fold}; test evaluated only after final checkpoint", flush=True)
