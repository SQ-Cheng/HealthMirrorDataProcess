"""Fine-tune a frozen pretrained ECG encoder for multi-task SQA.

Examples:
    python train_sqa.py --encoder tcn --pretrained-checkpoint ../checkpoints/exp1_ecg_tcn_L1024tcn01_best.pt
    python train_sqa.py --encoder resnet --pretrained-checkpoint ../checkpoints/exp1_ecg_resnet_L1024resnet01_best.pt
    python train_sqa.py --sanity-check
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.dataloader import MaskedReconDataset
from exp1_sqa_pretrain.models import build_resnet_encoder, build_tcn_encoder
from exp1_sqa_pretrain.sqa.model import ECGSQAModel, load_pretrained_encoder
from exp1_sqa_pretrain.sqa.weak_labels import (
    FEATURE_NAMES,
    ECGWeakLabelGenerator,
    WeakLabelConfig,
)


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(_PKG_DIR, "sqa_outputs")
DEFAULT_CHECKPOINT_DIR = os.path.join(_PKG_DIR, "checkpoints")
DEFAULT_REFERENCE_DIR = os.path.join(os.path.dirname(_STUDY_DIR), "reference_ecg")


class WeakLabelDataset(Dataset):
    """Attach precomputed weak SQA labels to the existing ECG windows."""

    def __init__(self, base_dataset, sampling_rate_hz, generator=None):
        self.base_dataset = base_dataset
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.generator = generator or ECGWeakLabelGenerator()
        self.weak_labels = []

        print(f"[SQA] Generating weak labels for {len(base_dataset)} ECG windows...")
        for index, sample in enumerate(base_dataset.samples):
            label = self.generator(sample[0], self.sampling_rate_hz)
            self.weak_labels.append(label)
            if (index + 1) % 5000 == 0:
                print(f"[SQA] Weak labels: {index + 1}/{len(base_dataset)}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        label = self.weak_labels[index]
        features = label["features"]
        return {
            "ecg": torch.from_numpy(self.base_dataset.samples[index]),
            "target": torch.tensor(
                [label["y_qrs"], label["y_morph"]], dtype=torch.float32
            ),
            "sample_weight": torch.tensor(label["sample_weight"], dtype=torch.float32),
            "features": torch.tensor(
                [features[name] for name in FEATURE_NAMES], dtype=torch.float32
            ),
            "dataset_index": torch.tensor(index, dtype=torch.long),
        }


def _split_grouped(group_values, val_ratio, seed):
    unique_groups = list(dict.fromkeys(group_values))
    if len(unique_groups) < 2:
        return None

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_val = max(1, int(round(len(unique_groups) * val_ratio)))
    n_val = min(n_val, len(unique_groups) - 1)
    val_groups = set(unique_groups[:n_val])

    train_indices = [i for i, group in enumerate(group_values) if group not in val_groups]
    val_indices = [i for i, group in enumerate(group_values) if group in val_groups]
    return train_indices, val_indices


def leakage_safe_split(dataset, val_ratio, seed):
    """Prefer patient, then session, then gap-separated contiguous splitting."""
    base = dataset.base_dataset

    split = _split_grouped(base.hospital_pids, val_ratio, seed)
    if split is not None:
        return *split, "patient"

    sessions = [record["file_path"] for record in base.source_records]
    split = _split_grouped(sessions, val_ratio, seed)
    if split is not None:
        return *split, "session"

    if len(dataset) < 2:
        raise RuntimeError("At least two windows are required for train/validation split.")

    ordered = sorted(
        range(len(dataset)),
        key=lambda i: base.source_records[i]["window_start_time"],
    )
    n_val = max(1, int(round(len(ordered) * val_ratio)))
    n_val = min(n_val, len(ordered) - 1)
    split_at = len(ordered) - n_val

    record = base.source_records[ordered[0]]
    gap_windows = int(math.ceil(record["window_sec"] / max(1e-8, record["step_samples"] / record["sampling_rate_hz"])))
    train_end = max(1, split_at - gap_windows)
    train_indices = ordered[:train_end]
    val_indices = ordered[split_at:]
    return train_indices, val_indices, "contiguous_with_gap"


def build_sqa_dataloaders(args, weak_config):
    base_dataset = MaskedReconDataset(
        args.data_dir,
        signal_type="ecg",
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )
    effective_fs = args.target_length / args.window_sec
    dataset = WeakLabelDataset(
        base_dataset,
        sampling_rate_hz=effective_fs,
        generator=ECGWeakLabelGenerator(
            weak_config,
            template_source=args.template_source,
            reference_dir=args.reference_ecg_dir,
        ),
    )
    train_indices, val_indices, split_mode = leakage_safe_split(
        dataset, args.val_ratio, args.seed
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(dataset, train_indices), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False, **loader_kwargs
    )
    print(
        f"[SQA] {split_mode} split: train={len(train_indices)}, "
        f"val={len(val_indices)}, effective_fs={effective_fs:.3f} Hz"
    )
    return train_loader, val_loader, dataset, split_mode


def weighted_multitask_loss(logits, targets, sample_weight, morph_loss_weight):
    per_task = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    combined = per_task[:, 0] + morph_loss_weight * per_task[:, 1]
    loss = (sample_weight * combined).mean()
    assert torch.isfinite(loss)
    return loss, per_task


def _binary_metrics(soft_targets, probabilities):
    labels = np.asarray(soft_targets) >= 0.5
    scores = np.asarray(probabilities, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan"), float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    auroc = (
        ranks[labels].sum() - positives * (positives + 1) / 2.0
    ) / (positives * negatives)

    descending = np.argsort(-scores, kind="stable")
    sorted_labels = labels[descending]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    auprc = float(precision[sorted_labels].sum() / positives)
    return float(auroc), auprc


def compute_metrics(losses, task_bce, targets, probabilities):
    targets = np.concatenate(targets, axis=0)
    probabilities = np.concatenate(probabilities, axis=0)
    task_bce = np.concatenate(task_bce, axis=0)

    qrs_auroc, qrs_auprc = _binary_metrics(targets[:, 0], probabilities[:, 0])
    morph_auroc, morph_auprc = _binary_metrics(targets[:, 1], probabilities[:, 1])
    metrics = {
        "loss": float(np.mean(losses)),
        "bce_qrs": float(np.mean(task_bce[:, 0])),
        "bce_morph": float(np.mean(task_bce[:, 1])),
        "auroc_qrs": qrs_auroc,
        "auroc_morph": morph_auroc,
        "auprc_qrs": qrs_auprc,
        "auprc_morph": morph_auprc,
        "mean_p_qrs": float(np.mean(probabilities[:, 0])),
        "mean_p_morph": float(np.mean(probabilities[:, 1])),
    }
    for threshold in (0.5, 0.8, 0.9):
        suffix = str(threshold).replace(".", "")
        metrics[f"accept_qrs_{suffix}"] = float(np.mean(probabilities[:, 0] >= threshold))
        metrics[f"accept_morph_{suffix}"] = float(np.mean(probabilities[:, 1] >= threshold))
    return metrics


def run_epoch(
    model,
    loader,
    device,
    morph_loss_weight,
    optimizer=None,
    max_batches=None,
    collect_diagnostics=False,
):
    training = optimizer is not None
    model.train(training)

    losses = []
    task_bce_values = []
    targets_values = []
    probability_values = []
    diagnostic_batches = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        ecg = batch["ecg"].to(device)
        targets = batch["target"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(ecg)
        loss, task_bce = weighted_multitask_loss(
            logits, targets, sample_weight, morph_loss_weight
        )

        if training:
            loss.backward()
            optimizer.step()

        probabilities = torch.sigmoid(logits)
        losses.append(float(loss.detach().cpu()))
        task_bce_values.append(task_bce.detach().cpu().numpy())
        targets_values.append(targets.detach().cpu().numpy())
        probability_values.append(probabilities.detach().cpu().numpy())

        if collect_diagnostics:
            diagnostic_batches.append({
                "indices": batch["dataset_index"].cpu().numpy(),
                "targets": targets.detach().cpu().numpy(),
                "weights": batch["sample_weight"].cpu().numpy(),
                "probabilities": probabilities.detach().cpu().numpy(),
                "features": batch["features"].cpu().numpy(),
            })

    if not losses:
        raise RuntimeError("No batches were processed. Check the split and batch limits.")

    metrics = compute_metrics(
        losses, task_bce_values, targets_values, probability_values
    )
    return metrics, diagnostic_batches


def _format_metric(value):
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def write_history(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics(diagnostic_batches, dataset, path, limit):
    rows = []
    for batch in diagnostic_batches:
        for row_index, dataset_index in enumerate(batch["indices"]):
            source = dataset.base_dataset.get_source_record(int(dataset_index))
            row = {
                "dataset_index": int(dataset_index),
                "hospital_patient_id": source["hospital_patient_id"],
                "file_name": source["file_name"],
                "window_start_time": source["window_start_time"],
                "y_qrs": float(batch["targets"][row_index, 0]),
                "y_morph": float(batch["targets"][row_index, 1]),
                "sample_weight": float(batch["weights"][row_index]),
                "p_qrs": float(batch["probabilities"][row_index, 0]),
                "p_morph": float(batch["probabilities"][row_index, 1]),
            }
            row.update({
                name: float(batch["features"][row_index, feature_index])
                for feature_index, name in enumerate(FEATURE_NAMES)
            })
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path, model, optimizer, args, weak_config, epoch, metrics, split_mode):
    torch.save({
        "task": "ecg_sqa",
        "encoder_architecture": args.encoder,
        "pretrained_checkpoint": os.path.abspath(args.pretrained_checkpoint),
        "target_length": args.target_length,
        "window_sec": args.window_sec,
        "effective_sampling_rate_hz": args.target_length / args.window_sec,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "split_mode": split_mode,
        "weak_label_config": asdict(weak_config),
        "training_config": vars(args),
    }, path)


def run_sanity_check():
    """Dependency-free synthetic test covering both supported encoders."""
    torch.manual_seed(7)
    np.random.seed(7)
    length = 512
    fs = 100.0
    time_axis = np.arange(length) / fs
    ecg = 0.03 * np.sin(2 * np.pi * 1.2 * time_axis)
    for peak in range(50, length, 83):
        indices = np.arange(length)
        ecg += np.exp(-0.5 * ((indices - peak) / 2.0) ** 2)
    ecg += 0.01 * np.random.randn(length)

    generator = ECGWeakLabelGenerator()
    labels = [generator(ecg, fs), generator(np.zeros_like(ecg), fs)]
    for label in labels:
        assert 0.0 <= label["y_qrs"] <= 1.0
        assert 0.0 <= label["y_morph"] <= 1.0
        assert 0.0 <= label["sample_weight"] <= 1.0
        assert all(np.isfinite(label["features"][name]) for name in FEATURE_NAMES)

    batch = torch.tensor(
        np.stack([ecg, np.zeros_like(ecg)])[:, None, :], dtype=torch.float32
    )
    targets = torch.tensor(
        [[label["y_qrs"], label["y_morph"]] for label in labels], dtype=torch.float32
    )
    weights = torch.tensor(
        [label["sample_weight"] for label in labels], dtype=torch.float32
    )

    builders = {
        "tcn": build_tcn_encoder,
        "resnet": build_resnet_encoder,
    }
    for name, builder in builders.items():
        model = ECGSQAModel(builder(target_length=length), freeze_encoder=True)
        captured_input = []

        first_conv = next(
            module for module in model.encoder.modules()
            if isinstance(module, torch.nn.Conv1d)
        )
        hook = first_conv.register_forward_pre_hook(
            lambda _module, inputs: captured_input.append(inputs[0].detach())
        )
        logits = model(batch)
        hook.remove()

        assert logits.shape == (2, 2)
        assert captured_input[0].shape[1] == 2
        assert torch.count_nonzero(captured_input[0][:, 1]).item() == 0
        loss, _ = weighted_multitask_loss(logits, targets, weights, 0.5)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
        assert all(parameter.grad is None for parameter in model.encoder.parameters())
        assert any(parameter.grad is not None for parameter in model.head.parameters())
        print(f"[sanity] {name}: logits={tuple(logits.shape)}, loss={loss.item():.4f}")

    print("[sanity] weak labels, zero-mask input, frozen encoders, and head gradients passed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-encoder multi-task ECG SQA fine-tuning"
    )
    parser.add_argument("--encoder", choices=["tcn", "resnet"], default="tcn")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default="/root/shared/HealthMirrorDataset")
    parser.add_argument("--data-source", choices=["sqi", "cleaned"], default="sqi")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for SQA checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-tag", type=str, default="",
        help="Optional tag appended to SQA checkpoint and output names.",
    )
    parser.add_argument(
        "--template-source",
        choices=["window", "reference"],
        default="window",
        help="Use the current window median beat or reference_ecg files.",
    )
    parser.add_argument(
        "--reference-ecg-dir",
        type=str,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing reference ECG CSV files.",
    )

    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--morph-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--diagnostic-limit", type=int, default=1000)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run a synthetic check for both encoders; no checkpoint or data required.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sanity_check:
        run_sanity_check()
        return
    if not args.pretrained_checkpoint:
        raise ValueError("--pretrained-checkpoint is required for fine-tuning")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_tag = args.checkpoint_tag.strip()
    if checkpoint_tag and (
        os.path.basename(checkpoint_tag) != checkpoint_tag
        or not all(character.isalnum() or character in {"-", "_"} for character in checkpoint_tag)
    ):
        raise ValueError("--checkpoint-tag may only contain letters, numbers, '-' and '_'")

    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    run_name = (
        f"exp1_sqa_ecg_{args.encoder}_{args.template_source}_"
        f"L{args.target_length}{tag_suffix}"
    )
    output_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    weak_config = WeakLabelConfig()

    encoder, pretrained_metadata = load_pretrained_encoder(
        args.encoder,
        args.pretrained_checkpoint,
        target_length=args.target_length,
        map_location="cpu",
    )
    model = ECGSQAModel(encoder, freeze_encoder=True).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader, val_loader, dataset, split_mode = build_sqa_dataloaders(
        args, weak_config
    )
    config = {
        "training": vars(args),
        "weak_labels": asdict(weak_config),
        "split_mode": split_mode,
        "effective_sampling_rate_hz": args.target_length / args.window_sec,
        "pretrained_metadata": {
            key: pretrained_metadata.get(key)
            for key in ("model", "signal_type", "target_length", "epoch", "val_loss")
        },
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_parameters)
    print(
        f"[SQA] encoder={args.encoder}, device={device}, "
        f"parameters={total_parameters:,}, trainable={trainable_count:,}"
    )

    history = []
    best_val_loss = float("inf")
    best_diagnostics = []
    best_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            args.morph_loss_weight,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_metrics, diagnostics = run_epoch(
                model,
                val_loader,
                device,
                args.morph_loss_weight,
                max_batches=args.max_val_batches,
                collect_diagnostics=True,
            )

        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"QRS AUC={_format_metric(val_metrics['auroc_qrs'])} "
            f"Morph AUC={_format_metric(val_metrics['auroc_morph'])} | "
            f"p=({val_metrics['mean_p_qrs']:.3f}, {val_metrics['mean_p_morph']:.3f})"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_diagnostics = diagnostics
            save_checkpoint(
                best_path,
                model,
                optimizer,
                args,
                weak_config,
                epoch,
                val_metrics,
                split_mode,
            )

    final_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    save_checkpoint(
        final_path,
        model,
        optimizer,
        args,
        weak_config,
        args.epochs,
        val_metrics,
        split_mode,
    )
    write_history(history, os.path.join(output_dir, "history.csv"))
    write_diagnostics(
        best_diagnostics,
        dataset,
        os.path.join(output_dir, "val_diagnostics.csv"),
        args.diagnostic_limit,
    )
    print(f"[SQA] best checkpoint: {best_path}")
    print(f"[SQA] final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
