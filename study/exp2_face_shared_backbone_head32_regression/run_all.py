"""Native224 ten-task shared-backbone ablation with a global patient split."""

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from . import config
from study.exp2_face_pretrained_head32_regression.data import _candidate_score, _distribution_audit, _plot_split_distributions
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler, fit_robust_target_scaler, write_target_scalers
from .shared_backbone import SharedBackboneModel, SharedFrameDataset, SharedLabBatchSampler, masked_losses, task_balanced_loss
from study.exp2_face_pretrained_head32_regression.train import _prepare_images, _regression_metrics


BASE = config.REFERENCE_OUTPUT_DIR
OUTPUT = config.OUTPUT_DIR
LOGS = config.LOG_DIR
TARGETS = config.ALL_REGRESSION_TARGETS


def execution_model(model):
    if not config.TORCH_COMPILE_ENABLED:
        return model, "eager"
    compiled = torch.compile(
        model, dynamic=config.TORCH_COMPILE_DYNAMIC,
        options={"triton.cudagraphs": config.TORCH_COMPILE_CUDAGRAPHS},
    )
    print("[compile-enabled] mode=default dynamic_batches=True cudagraphs=False; evaluation=eager", flush=True)
    return compiled, "inductor:default_dynamic_no_cudagraphs"


def record_memory_fix(resume):
    path = OUTPUT / "experiment_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["execution"] = {
        "backend": "inductor", "mode": "default", "dynamic_batches": True,
        "cudagraphs": False, "evaluation": "eager",
        "reason": "OOM at first finetune epoch; log reports 14.4 GiB of 15.3 GiB allocated in private CUDA Graph pools",
        "data_split_labels_loss_and_batch_unchanged": True,
    }
    if resume:
        manifest["restart"] = {
            "from": str(OUTPUT / "best_head.pt"), "stage": "finetune",
            "head_history_preserved": True, "completed_finetune_epochs": 0,
            "rng": "deterministic fresh seed; exact pre-crash CUDA RNG state was not saved",
        }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def select_shared_split(records):
    patient_values = []
    for target, table in records.items():
        values = table.groupby("hospital_id").raw_value.median().rank(pct=True).rename(target)
        patient_values.append(values)
    patient_table = pd.concat(patient_values, axis=1).sort_index()
    patients = patient_table.index.to_numpy(str)
    bins = pd.qcut(patient_table.mean(axis=1), 8, labels=False, duplicates="drop").to_numpy()
    lookup = {patient: index for index, patient in enumerate(patients)}
    row_indices = {target: table.hospital_id.map(lookup).to_numpy(int) for target, table in records.items()}
    best = best_passed = None
    for candidate in range(config.SPLIT_CANDIDATES):
        rng = np.random.default_rng(config.SEED + candidate * 104729)
        assignment = np.empty(len(patients), dtype=np.int8)
        for bucket in np.unique(bins):
            indices = rng.permutation(np.flatnonzero(bins == bucket))
            train = min(max(1, round(.6 * len(indices))), len(indices) - 2)
            val = min(max(1, round(.2 * len(indices))), len(indices) - train - 1)
            assignment[indices[:train]] = 0
            assignment[indices[train:train + val]] = 1
            assignment[indices[train + val:]] = 2
        scores = {target: _candidate_score(table, assignment, row_indices[target], ["raw_value", "abnormal_score"])
                  for target, table in records.items()}
        objectives = np.array([score["objective"] for score in scores.values()])
        key = (float(objectives.mean() + objectives.max()), candidate)
        entry = (key, assignment.copy(), scores, candidate)
        if best is None or key < best[0]: best = entry
        if all(score["passed"] for score in scores.values()) and (best_passed is None or key < best_passed[0]):
            best_passed = entry
    if best_passed is None:
        raise RuntimeError(f"No common balanced split passed {config.SPLIT_CANDIDATES} candidates; best={best[2]}")
    _, assignment, scores, candidate = best_passed
    mapping = dict(zip(patients, np.asarray(["train", "val", "test"])[assignment]))
    return mapping, {"selected_candidate": candidate, "candidate_count": config.SPLIT_CANDIDATES,
                     "objective": best_passed[0][0], "per_target_scores": scores,
                     "algorithm": "eight bins of mean within-analyte patient rank; minimize mean+max of inherited per-task distribution objectives"}


def build_union(records):
    videos = pd.concat([table[["hospital_id", "video_id", "mirror", "lab_patient_id", "split"]] for table in records.values()])
    if videos.groupby("video_id").hospital_id.nunique().gt(1).any() or videos.groupby("hospital_id").split.nunique().gt(1).any():
        raise ValueError("Shared encoder would leak patient identity across tasks")
    videos = videos.drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)
    labels = np.zeros((len(videos), len(TARGETS)), np.float32); masks = np.zeros_like(labels, dtype=bool)
    events = [set() for _ in range(len(videos))]
    lookup = dict(zip(videos.video_id, range(len(videos))))
    for column, target in enumerate(TARGETS):
        table = records[target]
        rows = table.video_id.map(lookup).to_numpy(int)
        labels[rows, column] = table.robust_scaled_raw_value.to_numpy(np.float32); masks[rows, column] = True
        for row, event in zip(rows, table.clinical_event_id):
            events[row].add(f"{target}|{event}")
    return videos, labels, masks, events


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "task_records").mkdir(exist_ok=True)
    if (OUTPUT / "model.pt").exists(): raise FileExistsError("Shared-backbone checkpoint already exists; not overwriting")
    reference = json.loads((BASE / "direct_field_extension_protocol.json").read_text())
    if sha256(Path(config.EXP_DIR).parents[1] / "merged_lab_tests.csv") != reference["lab_table_sha256"]:
        raise RuntimeError("Main reference is not based on the current lab table")
    records, hashes = {}, {}
    expected = {**reference["original_eight_records_sha256"], **reference["record_hashes"][str(BASE)]}
    for target in TARGETS:
        path = BASE / f"task_records/{target}.csv"; hashes[target] = sha256(path)
        if hashes[target] != expected[target]: raise RuntimeError(f"Main records changed: {target}")
        records[target] = pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
    old = pd.concat([table[["hospital_id", "split"]] for table in records.values()])
    conflicts = int(old.groupby("hospital_id").split.nunique().gt(1).sum())
    assignments, selection = select_shared_split(records)
    scalers, summaries, audits, pairs = {}, [], [], []
    for target, table in records.items():
        table["original_main_split"] = table.split
        table["split"] = table.hospital_id.map(assignments)
        scaler = fit_robust_target_scaler(target, table, config.SCORE_DEFINITIONS[target]["unit"])
        table["robust_scaled_raw_value"] = scaler.transform(table.raw_value)
        scalers[target] = scaler; table.to_csv(OUTPUT / f"task_records/{target}.csv", index=False)
        summary, pair = _distribution_audit(table, target); audits.extend(summary); pairs.extend(pair)
        row = {"target": target, "videos": len(table), "patients": table.hospital_id.nunique()}
        for split, group in table.groupby("split"):
            row.update({f"{split}_videos": len(group), f"{split}_patients": group.hospital_id.nunique()})
        summaries.append(row)
    videos, labels, masks, events = build_union(records)
    videos.to_csv(OUTPUT / "video_manifest.csv", index=False)
    pd.DataFrame({"hospital_id": assignments.keys(), "split": assignments.values()}).to_csv(OUTPUT / "patient_split.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "task_summary.csv", index=False)
    pd.DataFrame(audits).to_csv(OUTPUT / "split_distribution_audit.csv", index=False)
    pd.DataFrame(pairs).to_csv(OUTPUT / "split_distribution_pairwise.csv", index=False)
    write_target_scalers(scalers, OUTPUT / "target_scalers.json")
    _plot_split_distributions(records, OUTPUT)
    training = masks[videos.split.eq("train").to_numpy()]
    weights = len(training) / (len(TARGETS) * training.sum(axis=0))
    manifest = {
        "experiment": "exp2_face_shared_backbone_head32_regression",
        "output_directory": str(OUTPUT), "log_directory": str(LOGS),
        "targets": list(TARGETS), "lab_table_sha256": reference["lab_table_sha256"],
        "source_records_sha256": hashes, "frame_index": reference["frame_index"],
        "frame_index_sha256": reference["frame_index_sha256"],
        "source": str(BASE), "videos": len(videos), "patients": videos.hospital_id.nunique(),
        "original_cross_task_split_conflicts": conflicts, "split_selection": selection,
        "comparison": "only shared model trained; original-main split differs; full cohort and common held-out patients reported separately",
        "unchanged": ["all per-task videos and measured labels", "nearest24h matching", "native224 twenty frames", "five views",
                      "head32 architecture", "two-stage full fine-tuning", "AdamW", "learning rates and patience"],
        "training": {"head_lr": config.HEAD_LEARNING_RATE, "finetune_lr": config.FINETUNE_LEARNING_RATE,
                     "head_epochs": config.HEAD_MAX_EPOCHS, "finetune_epochs": config.FINETUNE_MAX_EPOCHS,
                     "head_patience": config.HEAD_PATIENCE, "finetune_patience": config.FINETUNE_PATIENCE,
                     "min_lr": config.MIN_LEARNING_RATE, "weight_decay": config.WEIGHT_DECAY,
                     "scheduler": "cosine, no warmup", "loss": "equal-task mean of observed-frame SmoothL1(beta=0.5)",
                     "task_sampling_correction_weights": dict(zip(TARGETS, weights.tolist())),
                     "missing_targets": "masked; no imputation", "frame_batch_size": 240,
                     "batch": "up to twelve video/view groups; no repeat target-specific assay event within a batch",
                     "selection": "one shared checkpoint by equal mean of video-level validation MAE / train IQR; no per-head encoder checkpoint mixing",
                     "epochs": "all selected frames/views and every observed target used once; no oversampling",
                     "device": "one joint model on cuda:0",
                     "execution": {"backend": "inductor", "mode": "default", "dynamic_batches": True,
                                   "cudagraphs": False, "evaluation": "eager"}},
        "record_hashes": {target: sha256(OUTPUT / f"task_records/{target}.csv") for target in TARGETS},
    }
    (OUTPUT / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    preflight()
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def preflight():
    manifest = json.loads((OUTPUT / "experiment_manifest.json").read_text())
    if sha256(Path(config.EXP_DIR).parents[1] / "merged_lab_tests.csv") != manifest["lab_table_sha256"]:
        raise RuntimeError("Lab table changed")
    index_path = Path(manifest["frame_index"]); index = FrameOffsetIndex.load(index_path)
    if sha256(index_path) != manifest["frame_index_sha256"] or not _index_is_reusable(index_path.parent, index.video_ids, "20frame"):
        raise RuntimeError("Native224 cache is stale")
    scalers = json.loads((OUTPUT / "target_scalers.json").read_text())["targets"]
    records = {}
    for target in TARGETS:
        assert sha256(BASE / f"task_records/{target}.csv") == manifest["source_records_sha256"][target]
        assert sha256(OUTPUT / f"task_records/{target}.csv") == manifest["record_hashes"][target]
        table = pd.read_csv(OUTPUT / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str}, float_precision="round_trip")
        assert not table.video_id.duplicated().any() and table.match_delta_h.between(0, 24 + 1e-9).all()
        assert fit_robust_target_scaler(target, table, config.SCORE_DEFINITIONS[target]["unit"]).to_dict() == scalers[target]
        records[target] = table
    videos, labels, masks, events = build_union(records)
    assert videos.groupby("hospital_id").split.nunique().eq(1).all()
    assert all(index.frame_range(video)[1] - index.frame_range(video)[0] == 20 for video in videos.video_id)
    np.testing.assert_array_equal(masks.sum(axis=0), [len(records[target]) for target in TARGETS])
    train_rows = np.flatnonzero(videos.split.eq("train"))
    sampler = SharedLabBatchSampler([events[row] for row in train_rows])
    coverage = np.sort(np.concatenate(sampler.batches))
    np.testing.assert_array_equal(coverage, np.arange(len(train_rows) * 100))
    for batch in sampler:
        used = set()
        for group in np.asarray(batch).reshape(-1, 20):
            assert len(set(group // 100)) == 1 and len(set(group % 5)) == 1
            selected = events[train_rows[group[0] // 100]]
            assert not used & selected; used.update(selected)
    print("[preflight-ok] ten unchanged 24h target cohorts; global patient-disjoint split; all masks/frames/views covered; train-only scaling", flush=True)
    return records, videos, labels, masks, events, index, scalers, manifest


def loaders(index, videos, labels, masks, events, workers=None):
    result, datasets = {}, {}
    for split in ("train", "val", "test"):
        rows = np.flatnonzero(videos.split.eq(split))
        datasets[split] = SharedFrameDataset(index, videos.iloc[rows], labels[rows], masks[rows], ("original",))
        count = config.EVAL_NUM_WORKERS if workers is None else workers
        result[split] = DataLoader(datasets[split], batch_size=512, num_workers=count, pin_memory=True,
                                   persistent_workers=count > 0, prefetch_factor=config.PREFETCH_FACTOR if count else None)
    rows = np.flatnonzero(videos.split.eq("train"))
    dataset = SharedFrameDataset(index, videos.iloc[rows], labels[rows], masks[rows], config.VIEW_NAMES)
    datasets["augmented"] = dataset
    count = config.TRAIN_NUM_WORKERS if workers is None else workers
    result["augmented"] = DataLoader(dataset, batch_sampler=SharedLabBatchSampler([events[row] for row in rows]),
                                     num_workers=count, pin_memory=True, persistent_workers=count > 0,
                                     prefetch_factor=config.PREFETCH_FACTOR if count else None)
    return datasets, result


def macro_scaled_mae(metrics, scalers):
    return float(np.mean([metrics[target]["mae"] / scalers[target]["iqr"] for target in TARGETS]))


@torch.no_grad()
def evaluate(model, loader, dataset, scalers, records, split, device):
    model.eval(); predictions = []; positions = []; sums = torch.zeros(len(TARGETS), device=device)
    counts = torch.zeros_like(sums)
    for images, labels, mask, rows, codes in loader:
        images = _prepare_images(images, codes, "bicubic", device)
        labels = labels.to(device); mask = mask.to(device)
        with torch.autocast("cuda", dtype=torch.float16): output = model(images)
        sums += masked_losses(output, labels, mask).sum(dim=0); counts += mask.sum(dim=0)
        predictions.append(output.float().cpu().numpy()); positions.append(rows.numpy())
    if counts.min() == 0: raise RuntimeError(f"A task has no {split} labels")
    losses = (sums / counts).cpu().numpy()
    frame_predictions = np.concatenate(predictions); frame_rows = np.concatenate(positions)
    video_rows = dataset.frame_video_rows[frame_rows]
    video_ids = dataset.video_records.iloc[video_rows].video_id.to_numpy()
    metrics, tables, frames = {}, {}, {}
    for column, target in enumerate(TARGETS):
        table = records[target].loc[records[target].split.eq(split)]
        valid = dataset.task_masks[video_rows, column]
        aggregate = pd.DataFrame({"video_id": video_ids[valid], "prediction": frame_predictions[valid, column]}).groupby("video_id").agg(
            y_pred_scaled=("prediction", "mean"), frame_count=("prediction", "size"))
        merged = table.merge(aggregate, on="video_id", how="left", validate="one_to_one")
        assert len(merged) == len(table) and merged.frame_count.eq(20).all()
        scaler = RobustTargetScaler(**scalers[target])
        merged["y_true"] = merged.raw_value; merged["y_pred"] = scaler.inverse_transform(merged.y_pred_scaled)
        merged["y_true_scaled"] = merged.robust_scaled_raw_value; merged["residual"] = merged.y_pred - merged.y_true
        merged["architecture"] = "efficientnet_b0"; merged["target"] = target
        metrics[target] = _regression_metrics(merged.y_true, merged.y_pred, merged.score_threshold, config.SCORE_DEFINITIONS[target]["direction"])
        metrics[target]["frame_loss"] = float(losses[column]); tables[target] = merged
        raw_lookup = table.set_index("video_id").raw_value
        frames[target] = {
            "split": np.full(int(valid.sum()), split), "video_id": video_ids[valid],
            "source_frame_index": dataset.index.source_indices[dataset.frame_indices[frame_rows[valid]]],
            "y_true": raw_lookup.loc[video_ids[valid]].to_numpy(float),
            "y_pred": scaler.inverse_transform(frame_predictions[valid, column]),
        }
    return float(losses.mean()), metrics, tables, frames


def fit(records, videos, labels, masks, events, index, scalers, destination=OUTPUT,
        head_epochs=config.HEAD_MAX_EPOCHS, fine_epochs=config.FINETUNE_MAX_EPOCHS, smoke=False,
        compile_smoke=False, resume_finetune=False):
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    torch.cuda.set_device(0); torch.cuda.manual_seed_all(config.SEED); torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    datasets, data = loaders(index, videos, labels, masks, events, workers=0 if smoke else None)
    model = SharedBackboneModel(TARGETS).to(device, memory_format=torch.channels_last)
    train_mask = masks[videos.split.eq("train").to_numpy()]
    weights = torch.tensor(len(train_mask) / (len(TARGETS) * train_mask.sum(axis=0)), dtype=torch.float32, device=device)
    backbone_params = sum(p.numel() for p in model.features.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())
    print(f"[job-start] shared_encoder={backbone_params} heads={head_params} per_head={head_params//len(TARGETS)} "
          f"targets={len(TARGETS)} videos={len(videos)} device={device} frame_batch=240 masked_task_balanced_loss", flush=True)
    joint_history, target_history = [], {target: [] for target in TARGETS}
    best_overall = None
    stages = (
        ("head", head_epochs, config.HEAD_LEARNING_RATE, config.HEAD_PATIENCE),
        ("finetune", fine_epochs, config.FINETUNE_LEARNING_RATE, config.FINETUNE_PATIENCE),
    )
    if resume_finetune:
        saved = torch.load(destination / "best_head.pt", map_location="cpu", weights_only=True)
        if tuple(saved["targets"]) != TARGETS or saved["target_scalers"] != scalers:
            raise RuntimeError("Head checkpoint does not match the prepared targets/scalers")
        joint_history = pd.read_csv(destination / "shared_training_history.csv").to_dict("records")
        if not joint_history or any(row["stage"] != "head" for row in joint_history):
            raise RuntimeError("This recovery requires completed head-only history")
        for target in TARGETS:
            target_history[target] = pd.read_csv(destination / f"runs/efficientnet_b0/{target}/history.csv").to_dict("records")
            if len(target_history[target]) != len(joint_history):
                raise RuntimeError("Per-task head histories are incomplete")
        model.load_state_dict(saved["model_state_dict"])
        best_overall = (saved["val_macro_scaled_mae"], "head", saved["model_state_dict"])
        stages = stages[1:]
        print(f"[resume] preserved head_epochs={len(joint_history)}; best_head_epoch={saved['epoch']}; start finetune", flush=True)
    for stage, epochs, lr, patience_limit in stages:
        if stage == "head": model.freeze_encoder()
        else:
            for p in model.parameters(): p.requires_grad = True
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=config.MIN_LEARNING_RATE)
        amp = torch.amp.GradScaler("cuda", init_scale=1024)
        execution, backend = (model, "eager") if smoke and not compile_smoke else execution_model(model)
        best_state, best_score, patience = None, float("inf"), 0
        print(f"[stage-start] {stage} epochs={epochs} lr={lr:g} patience={patience_limit}", flush=True)
        for epoch in range(1, epochs + 1):
            if stage == "head": model.eval(); model.heads.train()
            else: model.train()
            data["augmented"].batch_sampler.set_epoch(len(joint_history) + 1)
            sums = torch.zeros(len(TARGETS), device=device); counts = torch.zeros_like(sums)
            loss_sum = inputs = 0; started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            for batch, (images, truth, mask, _, codes) in enumerate(data["augmented"]):
                if smoke and batch >= 1: break
                images = _prepare_images(images, codes, "bicubic", device)
                truth = truth.to(device); mask = mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = execution(images); loss = task_balanced_loss(prediction, truth, mask, weights)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite shared loss")
                amp.scale(loss).backward(); amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                amp.step(optimizer); amp.update()
                sums += masked_losses(prediction.detach(), truth, mask).sum(dim=0)
                counts += mask.sum(dim=0); loss_sum += float(loss.detach()) * len(images); inputs += len(images)
                del prediction, loss, images, truth, mask
            if not smoke:
                assert inputs == len(datasets["augmented"]), "Training did not retain every frame/view"
                np.testing.assert_array_equal(counts.cpu().numpy(), train_mask.sum(axis=0) * 100)
            optimization = (sums / counts.clamp_min(1)).cpu().numpy()
            train_loss, training, _, _ = evaluate(model, data["train"], datasets["train"], scalers, records, "train", device)
            val_loss, validation, _, _ = evaluate(model, data["val"], datasets["val"], scalers, records, "val", device)
            val_score = macro_scaled_mae(validation, scalers)
            global_epoch = len(joint_history) + 1
            base = {"stage": stage, "stage_epoch": epoch, "global_epoch": global_epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"], "train_model_inputs": inputs, "execution_backend": backend,
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
            joint_history.append({**base, "train_optimization_loss": loss_sum / inputs,
                                  "train_macro_loss": train_loss, "val_macro_loss": val_loss,
                                  "val_macro_scaled_mae": val_score})
            pd.DataFrame(joint_history).to_csv(destination / "shared_training_history.csv", index=False)
            for column, target in enumerate(TARGETS):
                row = {**base, "architecture": "efficientnet_b0", "target": target,
                       "train_loss": float(optimization[column]), "train_eval_loss": training[target]["frame_loss"],
                       "val_loss": validation[target]["frame_loss"],
                       **{f"train_{key}": value for key, value in training[target].items()},
                       **{f"val_{key}": value for key, value in validation[target].items()}}
                target_history[target].append(row)
                run = destination / f"runs/efficientnet_b0/{target}"; run.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(target_history[target]).to_csv(run / "history.csv", index=False)
            improved = val_score < best_score - 1e-5
            if improved:
                best_score = val_score; patience = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({"model_state_dict": best_state, "stage": stage, "epoch": epoch, "val_macro_loss": val_loss,
                            "val_macro_scaled_mae": val_score,
                            "targets": TARGETS, "target_scalers": scalers}, destination / f"best_{stage}.pt")
            else: patience += 1
            print(f"[epoch] {stage} {epoch:03}/{epochs} train_macro_loss={train_loss:.5f} val_macro_loss={val_loss:.5f} "
                  f"val_macro_scaled_MAE={val_score:.5f} "
                  f"train_r={np.nanmean([v['pearson_r'] for v in training.values()]):.4f} "
                  f"val_r={np.nanmean([v['pearson_r'] for v in validation.values()]):.4f} "
                  f"inputs={inputs} throughput={inputs/(time.monotonic()-started):.1f}/s "
                  f"mem={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GiB patience={patience}/{patience_limit}"
                  f"{'*' if improved else ''}", flush=True)
            scheduler.step()
            if patience >= patience_limit: break
        if best_state is None: raise RuntimeError("No finite shared checkpoint")
        model.load_state_dict(best_state)
        if best_overall is None or best_score < best_overall[0]: best_overall = (best_score, stage, best_state)
        model.zero_grad(set_to_none=True)
        del execution, optimizer, amp
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    model.load_state_dict(best_overall[2])
    torch.save({"architecture": "efficientnet_b0_shared_backbone_separate_head32", "targets": TARGETS,
                "model_state_dict": best_overall[2], "target_scalers": scalers, "selected_stage": best_overall[1],
                "val_macro_scaled_mae": best_overall[0], "backbone_parameters": backbone_params, "all_head_parameters": head_params,
                "task_type": "shared_backbone_raw_value_regression"}, destination / "model.pt")
    metric_rows, predictions, frames = [], {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    for split in ("train", "val", "test"):
        _, results, tables, compact = evaluate(model, data[split], datasets[split], scalers, records, split, device)
        for target in TARGETS:
            metric_rows.append({"architecture": "efficientnet_b0", "target": target, "split": split,
                                "selected_stage": best_overall[1], **results[target]})
            predictions[target].append(tables[target]); frames[target].append(compact[target])
    metrics = pd.DataFrame(metric_rows); metrics.to_csv(destination / "metrics_all.csv", index=False)
    rows = []
    for target in TARGETS:
        run = destination / f"runs/efficientnet_b0/{target}"
        metrics.loc[metrics.target.eq(target)].to_csv(run / "metrics.csv", index=False)
        pd.concat(predictions[target]).to_csv(run / "video_predictions.csv", index=False)
        np.savez_compressed(run / "frame_predictions.npz", **{key: np.concatenate([f[key] for f in frames[target]]) for key in frames[target][0]})
        torch.save({"target": target, "head_state_dict": model.heads[target].state_dict(),
                    "shared_checkpoint": str((destination / "model.pt").resolve())}, run / "head.pt")
        (run / "run_manifest.json").write_text(json.dumps({"target": target, "shared_checkpoint": str((destination / "model.pt").resolve()),
                                                        "selected_stage": best_overall[1]}, indent=2) + "\n")
        rows.append({"architecture": "efficientnet_b0", "target": target, "status": "ok"})
    pd.concat([pd.DataFrame(target_history[t]) for t in TARGETS]).to_csv(destination / "history_all.csv", index=False)
    pd.DataFrame(rows).to_csv(destination / "run_index.csv", index=False)
    for dataset in datasets.values(): dataset.close()
    print(f"[training-complete] selected={best_overall[1]} macro_val_scaled_MAE={best_overall[0]:.5f}", flush=True)


def smoke(compiled=False):
    records, videos, labels, masks, events, index, scalers, _ = preflight()
    keep = []
    for split in ("train", "val", "test"):
        candidates = videos.loc[videos.split.eq(split)].drop_duplicates("hospital_id").index.to_numpy()
        chosen = sorted(candidates, key=lambda row: -int(masks[row].sum()))[:12 if split == "train" else 4]
        if not masks[chosen].any(axis=0).all(): raise RuntimeError("Smoke does not cover all ten heads")
        keep.extend(chosen)
    subset = videos.iloc[keep].reset_index(drop=True)
    filtered = {target: table.loc[table.video_id.isin(subset.video_id)] for target, table in records.items()}
    with tempfile.TemporaryDirectory(prefix="shared_encoder_smoke_") as temporary:
        fit(filtered, subset, labels[keep], masks[keep], [events[row] for row in keep], index, scalers,
            Path(temporary), head_epochs=1, fine_epochs=1, smoke=True, compile_smoke=compiled)
        history = pd.read_csv(Path(temporary) / "shared_training_history.csv")
        if compiled:
            assert history.execution_backend.eq("inductor:default_dynamic_no_cudagraphs").all()
            assert history.peak_gpu_allocated_gib.max() < 14
            print(f"[memory-smoke] peak_allocated={history.peak_gpu_allocated_gib.max():.2f}GiB "
                  f"peak_reserved={history.peak_gpu_reserved_gib.max():.2f}GiB", flush=True)
        saved = torch.load(Path(temporary) / "model.pt", map_location="cpu", weights_only=True)
        assert len(saved["targets"]) == 10 and saved["all_head_parameters"] == 410890
        from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
        from .plot_comparison import plot_comparison
        plot(temporary)
        plot_comparison(BASE, temporary)
        assert len(list((Path(temporary) / "figures").glob("*.png"))) == 6
    torch.cuda.empty_cache()
    print("[smoke-ok] one encoder, ten heads, all target masks, two training stages, saved shared checkpoint", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--compiled-smoke", action="store_true")
    parser.add_argument("--resume-finetune", action="store_true")
    args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.reuse_prepared: prepare()
        if args.prepare_only: return
        if args.smoke or args.compiled_smoke: smoke(compiled=args.compiled_smoke); return
        if (OUTPUT / "model.pt").exists(): raise FileExistsError("Existing shared results will not be overwritten")
        records, videos, labels, masks, events, index, scalers, _ = preflight()
        record_memory_fix(args.resume_finetune)
        fit(records, videos, labels, masks, events, index, scalers, resume_finetune=args.resume_finetune)
        from unittest.mock import patch
        from study.exp2_face_pretrained_head32_regression import plot_results
        with patch.object(plot_results, "EXPERIMENT_LABEL", "Shared-backbone face regression (ten separate head32)"):
            plot_results.main(OUTPUT)
        from .plot_comparison import plot_comparison
        plot_comparison(BASE, OUTPUT)
        preflight()
        (OUTPUT / "COMPLETE").write_text("shared backbone, ten separate regression heads, metrics and figures completed\n")


if __name__ == "__main__":
    main()
