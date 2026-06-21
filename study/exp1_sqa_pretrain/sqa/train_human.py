"""Fine-tune window-template SQA heads with patient-disjoint human labels."""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.model import load_sqa_checkpoint
from exp1_sqa_pretrain.sqa.raw_windows import extract_window


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUEUE = os.path.join(
    _PKG_DIR, "sqa_annotations", "round01", "queue.csv"
)
DEFAULT_LABELS = os.path.join(
    _PKG_DIR, "sqa_annotations", "round01", "labels.csv"
)
DEFAULT_CHECKPOINT_DIR = os.path.join(_PKG_DIR, "checkpoints")
DEFAULT_OUTPUT_DIR = os.path.join(_PKG_DIR, "sqa_outputs", "human_finetune")
LABEL_MAP = {"good": 1.0, "bad": 0.0, "uncertain": np.nan}


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_task_label(values):
    mapped = [LABEL_MAP[value] for value in values if value in LABEL_MAP]
    definite = {value for value in mapped if np.isfinite(value)}
    if len(definite) == 1:
        return definite.pop(), False
    if len(definite) > 1:
        return np.nan, True
    return np.nan, False


def load_consolidated_labels(queue_path, labels_path, min_completed=80):
    queue = pd.read_csv(queue_path, dtype={"queue_id": str})
    labels = pd.read_csv(
        labels_path, dtype=str, keep_default_na=False
    )
    labels = labels[labels["status"] == "complete"].copy()
    if len(labels) < min_completed:
        raise RuntimeError(
            f"Only {len(labels)} completed tasks; require at least {min_completed}."
        )

    merged = queue.merge(labels, on="queue_id", how="inner", validate="one_to_one")
    rows = []
    conflicts = []
    required = {"overall_label", "qrs_override", "morph_override"}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"Labels use an incompatible schema; missing {sorted(missing)}")

    for window_id, group in merged.groupby("window_id", sort=False):
        qrs_values = [
            override if override in LABEL_MAP else overall
            for overall, override in zip(group["overall_label"], group["qrs_override"])
        ]
        morph_values = [
            override if override in LABEL_MAP else overall
            for overall, override in zip(group["overall_label"], group["morph_override"])
        ]
        qrs, qrs_conflict = _resolve_task_label(qrs_values)
        morph, morph_conflict = _resolve_task_label(morph_values)
        source = group.iloc[0]
        row = {
            "window_id": int(window_id),
            "mirror": source["mirror"],
            "patient_id": source["patient_id"],
            "patient_key": source["patient_key"],
            "file_path": source["file_path"],
            "start_time": float(source["start_time"]),
            "data_source": source.get("data_source", "raw"),
            "split": source["split"],
            "acquisition_type": source["acquisition_type"],
            "artifact_subtype": source.get("artifact_subtype", ""),
            "behavior_group": source.get("behavior_group", source["acquisition_type"]),
            "y_qrs": qrs,
            "y_morph": morph,
            "qrs_conflict": qrs_conflict,
            "morph_conflict": morph_conflict,
            "annotation_count": len(group),
        }
        rows.append(row)
        if qrs_conflict or morph_conflict:
            conflicts.append(row)

    frame = pd.DataFrame(rows)
    if not len(frame):
        raise RuntimeError("No usable completed annotations.")
    return frame, pd.DataFrame(conflicts, columns=frame.columns)


class HumanSQADataset(Dataset):
    def __init__(self, frame, window_sec, target_length):
        self.frame = frame.reset_index(drop=True).copy()
        inputs = []
        for _, row in self.frame.iterrows():
            window = extract_window(
                row["file_path"],
                row["start_time"],
                window_sec=window_sec,
                target_length=target_length,
                data_source=row.get("data_source", "raw"),
            )
            inputs.append(window["model_input"])
        self.inputs = np.stack(inputs)[:, None, :].astype(np.float32)
        self.targets = self.frame[["y_qrs", "y_morph"]].to_numpy(np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return {
            "ecg": torch.from_numpy(self.inputs[index]),
            "target": torch.from_numpy(self.targets[index]),
            "row_index": index,
        }


def _task_pos_weights(frame):
    weights = []
    for column in ("y_qrs", "y_morph"):
        values = frame[column].dropna().to_numpy()
        positives = int(np.sum(values == 1.0))
        negatives = int(np.sum(values == 0.0))
        weight = negatives / positives if positives and negatives else 1.0
        weights.append(float(np.clip(weight, 0.25, 4.0)))
    return weights


def masked_human_loss(logits, targets, pos_weights, morph_loss_weight):
    losses = []
    task_losses = []
    for task_index in range(2):
        valid = torch.isfinite(targets[:, task_index])
        if not torch.any(valid):
            task_losses.append(None)
            continue
        task_loss = F.binary_cross_entropy_with_logits(
            logits[valid, task_index],
            targets[valid, task_index],
            pos_weight=torch.tensor(
                pos_weights[task_index],
                device=logits.device,
                dtype=logits.dtype,
            ),
        )
        task_losses.append(task_loss)
        weight = 1.0 if task_index == 0 else morph_loss_weight
        losses.append(weight * task_loss)

    if not losses:
        raise RuntimeError("Batch contains no definite human labels.")
    loss = sum(losses) / sum(
        1.0 if index == 0 else morph_loss_weight
        for index, value in enumerate(task_losses)
        if value is not None
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite human-label loss.")
    return loss, task_losses


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return float("nan"), float("nan")

    order = np.argsort(scores)
    sorted_scores = np.asarray(scores)[order]
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

    descending = np.argsort(-np.asarray(scores), kind="stable")
    sorted_labels = labels[descending]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    auprc = precision[sorted_labels].sum() / positives
    return float(auroc), float(auprc)


def _classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (
        2.0 * precision * sensitivity / (precision + sensitivity)
        if np.isfinite(precision + sensitivity) and precision + sensitivity > 0
        else float("nan")
    )
    balanced = (
        0.5 * (sensitivity + specificity)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )
    auroc, auprc = _binary_auc(labels, probabilities)
    return {
        "n": len(labels),
        "positive_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _best_threshold(labels, probabilities):
    best_threshold, best_score = 0.5, -np.inf
    for threshold in np.linspace(0.05, 0.95, 91):
        score = _classification_metrics(
            labels, probabilities, threshold
        )["balanced_accuracy"]
        if np.isfinite(score) and score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


def _collect_metrics(targets, probabilities, thresholds=None):
    thresholds = thresholds or {"qrs": 0.5, "morph": 0.5}
    output = {}
    for task_index, task in enumerate(("qrs", "morph")):
        valid = np.isfinite(targets[:, task_index])
        if not np.any(valid):
            output[task] = {"n": 0}
            continue
        output[task] = _classification_metrics(
            targets[valid, task_index],
            probabilities[valid, task_index],
            thresholds[task],
        )
    return output


def run_epoch(model, loader, device, pos_weights, morph_loss_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses, all_targets, all_probabilities, all_indices = [], [], [], []

    for batch in loader:
        ecg = batch["ecg"].to(device)
        targets = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(ecg)
        loss, _ = masked_human_loss(
            logits, targets, pos_weights, morph_loss_weight
        )
        if training:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_indices.append(batch["row_index"].numpy())

    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)
    indices = np.concatenate(all_indices)
    metrics = _collect_metrics(targets, probabilities)
    metrics["loss"] = float(np.mean(losses))
    return metrics, targets, probabilities, indices


def _flatten_metrics(prefix, metrics):
    row = {f"{prefix}_loss": metrics["loss"]}
    for task in ("qrs", "morph"):
        for key in ("n", "auroc", "auprc", "balanced_accuracy", "brier"):
            row[f"{prefix}_{task}_{key}"] = metrics[task].get(key, float("nan"))
    return row


def _write_history(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _ordered_outputs(outputs):
    targets, probabilities, indices = outputs
    order = np.argsort(indices)
    return targets[order], probabilities[order]


def _category_summary(predictions):
    rows = []
    for group_column in ("acquisition_type", "behavior_group"):
        for group_name, group in predictions.groupby(group_column, dropna=False):
            if not str(group_name) or str(group_name) == "nan":
                continue
            for task in ("qrs", "morph"):
                target = group[f"y_{task}"].to_numpy(float)
                before = group[f"before_p_{task}"].to_numpy(float)
                after = group[f"after_p_{task}"].to_numpy(float)
                valid = np.isfinite(target)
                row = {
                    "group_type": group_column,
                    "group": group_name,
                    "task": task,
                    "n": len(group),
                    "labeled_n": int(valid.sum()),
                    "before_mean": float(before.mean()),
                    "after_mean": float(after.mean()),
                    "mean_delta": float((after - before).mean()),
                }
                if valid.any():
                    row["before_brier"] = float(np.mean((before[valid] - target[valid]) ** 2))
                    row["after_brier"] = float(np.mean((after[valid] - target[valid]) ** 2))
                else:
                    row["before_brier"] = np.nan
                    row["after_brier"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_category_changes(summary, path):
    subset = summary[summary["group_type"] == "acquisition_type"]
    groups = list(dict.fromkeys(subset["group"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(len(groups))
    for axis, task in zip(axes, ("qrs", "morph")):
        rows = subset[subset["task"] == task].set_index("group").reindex(groups)
        axis.plot(x, rows["before_mean"], "o-", label="before")
        axis.plot(x, rows["after_mean"], "o-", label="after")
        axis.set_xticks(x, groups, rotation=35, ha="right")
        axis.set_title(task.upper())
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean predicted reliability")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_checkpoint(
    path,
    model,
    optimizer,
    source_checkpoint,
    args,
    epoch,
    val_metrics,
    thresholds=None,
):
    torch.save({
        "task": "ecg_sqa_human",
        "encoder_architecture": args.encoder,
        "template_source": "window",
        "target_length": args.target_length,
        "window_sec": args.window_sec,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "head_state_dict": model.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "source_sqa_checkpoint": os.path.abspath(source_checkpoint),
        "queue_path": os.path.abspath(args.queue),
        "queue_sha256": _sha256(args.queue),
        "labels_path": os.path.abspath(args.labels),
        "labels_sha256": _sha256(args.labels),
        "validation_metrics": val_metrics,
        "thresholds": thresholds,
        "training_config": vars(args),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a window-template SQA head with human labels"
    )
    parser.add_argument("--encoder", choices=["tcn", "resnet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-tag", default="round01")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--morph-loss-weight", type=float, default=1.0)
    parser.add_argument("--min-completed", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    model, source_metadata = load_sqa_checkpoint(
        args.checkpoint, map_location="cpu", freeze_encoder=True
    )
    source_architecture = source_metadata.get("encoder_architecture")
    source_template = source_metadata.get("training_config", {}).get(
        "template_source",
        source_metadata.get("template_source"),
    )
    if source_architecture != args.encoder:
        raise ValueError(
            f"Checkpoint encoder is {source_architecture}, expected {args.encoder}."
        )
    if source_template != "window":
        raise ValueError("Human fine-tuning currently accepts window-template models only.")
    if int(source_metadata["target_length"]) != args.target_length:
        raise ValueError("Checkpoint target length does not match --target-length.")

    labels, conflicts = load_consolidated_labels(
        args.queue, args.labels, args.min_completed
    )
    usable = np.isfinite(labels[["y_qrs", "y_morph"]]).any(axis=1)
    labels = labels[usable].reset_index(drop=True)
    split_frames = {
        "train": labels[labels["split"] == "train"],
        "val": labels[labels["split"] == "val"],
        "test_random": labels[labels["split"] == "test_random"],
        "test_challenge": labels[labels["split"] == "test_challenge"],
    }
    for split, frame in split_frames.items():
        if not len(frame):
            raise RuntimeError(f"No usable labels for split: {split}")

    datasets = {
        split: HumanSQADataset(frame, args.window_sec, args.target_length)
        for split, frame in split_frames.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }

    model = model.to(device)
    before_outputs = {}
    with torch.no_grad():
        for split, loader in loaders.items():
            _, targets, probabilities, indices = run_epoch(
                model, loader, device, _task_pos_weights(split_frames["train"]),
                args.morph_loss_weight,
            )
            before_outputs[split] = (targets, probabilities, indices)

    optimizer = AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weights = _task_pos_weights(split_frames["train"])

    tag = args.checkpoint_tag.strip()
    if tag and not all(character.isalnum() or character in {"-", "_"} for character in tag):
        raise ValueError("Invalid --checkpoint-tag")
    suffix = f"_{tag}" if tag else ""
    run_name = (
        f"exp1_sqa_human_{args.encoder}_window_"
        f"L{args.target_length}{suffix}"
    )
    output_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
    final_path = os.path.join(args.checkpoint_dir, f"{run_name}_final.pt")

    conflicts.to_csv(os.path.join(output_dir, "annotation_conflicts.csv"), index=False)
    labels.to_csv(os.path.join(output_dir, "consolidated_labels.csv"), index=False)

    history = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics, _, _, _ = run_epoch(
            model,
            loaders["train"],
            device,
            pos_weights,
            args.morph_loss_weight,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_metrics, _, _, _ = run_epoch(
                model,
                loaders["val"],
                device,
                pos_weights,
                args.morph_loss_weight,
            )

        row = {"epoch": epoch, "seconds": time.time() - started}
        row.update(_flatten_metrics("train", train_metrics))
        row.update(_flatten_metrics("val", val_metrics))
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f}"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path,
                model,
                optimizer,
                args.checkpoint,
                args,
                epoch,
                val_metrics,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    _write_history(history, os.path.join(output_dir, "history.csv"))
    _save_checkpoint(
        final_path,
        model,
        optimizer,
        args.checkpoint,
        args,
        history[-1]["epoch"],
        val_metrics,
    )
    model, _ = load_sqa_checkpoint(
        best_path, map_location="cpu", freeze_encoder=True
    )
    model = model.to(device).eval()

    after_outputs = {}
    with torch.no_grad():
        for split, loader in loaders.items():
            _, targets, probabilities, indices = run_epoch(
                model, loader, device, pos_weights, args.morph_loss_weight,
            )
            after_outputs[split] = (targets, probabilities, indices)

    val_targets, val_probabilities = _ordered_outputs(after_outputs["val"])
    thresholds = {}
    for task_index, task in enumerate(("qrs", "morph")):
        valid = np.isfinite(val_targets[:, task_index])
        thresholds[task] = _best_threshold(
            val_targets[valid, task_index], val_probabilities[valid, task_index]
        )

    results = {"thresholds": thresholds, "splits": {}}
    prediction_frames = []
    for split in split_frames:
        before_targets, before_probabilities = _ordered_outputs(before_outputs[split])
        targets, after_probabilities = _ordered_outputs(after_outputs[split])
        if not np.allclose(before_targets, targets, equal_nan=True):
            raise RuntimeError(f"Target ordering changed for split {split}.")
        results["splits"][split] = {
            "before": _collect_metrics(targets, before_probabilities, thresholds),
            "after": _collect_metrics(targets, after_probabilities, thresholds),
        }
        source = datasets[split].frame.reset_index(drop=True).copy()
        source["before_p_qrs"] = before_probabilities[:, 0]
        source["before_p_morph"] = before_probabilities[:, 1]
        source["after_p_qrs"] = after_probabilities[:, 0]
        source["after_p_morph"] = after_probabilities[:, 1]
        prediction_frames.append(source)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    category_summary = _category_summary(predictions)
    with open(os.path.join(output_dir, "before_after_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(_json_safe(results), file, indent=2)
    predictions.to_csv(os.path.join(output_dir, "before_after_predictions.csv"), index=False)
    category_summary.to_csv(os.path.join(output_dir, "before_after_categories.csv"), index=False)
    _plot_category_changes(
        category_summary, os.path.join(output_dir, "before_after_categories.png")
    )
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    best_payload["thresholds"] = thresholds
    best_payload["human_evaluation"] = results["splits"]
    torch.save(best_payload, best_path)
    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
