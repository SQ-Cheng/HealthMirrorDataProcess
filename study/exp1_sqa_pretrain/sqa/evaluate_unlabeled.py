"""Evaluate trained SQA classifiers on unlabeled, uncleaned ECG recordings.

This script intentionally does not report accuracy/AUROC on raw data because no
human labels exist. It reports score distributions, acceptance rates, model
agreement, raw-artifact associations, and controlled-corruption sensitivity.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.model import load_sqa_checkpoint
from exp1_sqa_pretrain.sqa.raw_windows import load_raw_windows


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(_PKG_DIR, "sqa_outputs", "unlabeled_evaluation")
DEFAULT_DATA_ROOT = "/root/shared/HealthMirrorDataset"
TASKS = ("qrs", "morph")
SEVERITY_NAMES = ("mild", "moderate", "severe")
CORRUPTIONS = ("gaussian", "baseline", "impulse", "clipping", "dropout")


def _model_identity(checkpoint):
    architecture = checkpoint["encoder_architecture"]
    training_config = checkpoint.get("training_config", {})
    template_source = training_config.get(
        "template_source", checkpoint.get("template_source", "unknown")
    )
    return f"{architecture}_{template_source}"


def load_sqa_model(checkpoint_path, device):
    model, checkpoint = load_sqa_checkpoint(
        checkpoint_path, map_location="cpu", freeze_encoder=True
    )
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict(model, inputs, batch_size, device):
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        outputs.append(model.predict_proba(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def apply_corruption(inputs, corruption, severity_index, seed):
    rng = np.random.default_rng(seed)
    corrupted = inputs.copy()
    count, _, length = corrupted.shape

    if corruption == "gaussian":
        sigma = (0.25, 0.50, 1.00)[severity_index]
        corrupted += rng.normal(0.0, sigma, size=corrupted.shape).astype(np.float32)

    elif corruption == "baseline":
        amplitude = (0.50, 1.00, 2.00)[severity_index]
        time_axis = np.linspace(0.0, 1.0, length, endpoint=False)
        frequency = rng.uniform(0.5, 2.0, size=count)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=count)
        wave = np.sin(
            2.0 * np.pi * frequency[:, None] * time_axis[None, :] + phase[:, None]
        )
        corrupted[:, 0, :] += amplitude * wave.astype(np.float32)

    elif corruption == "impulse":
        impulse_count = (1, 3, 8)[severity_index]
        amplitude = (3.0, 6.0, 12.0)[severity_index]
        for sample_index in range(count):
            locations = rng.choice(length, size=impulse_count, replace=False)
            signs = rng.choice((-1.0, 1.0), size=impulse_count)
            corrupted[sample_index, 0, locations] += amplitude * signs

    elif corruption == "clipping":
        threshold = (2.0, 1.0, 0.5)[severity_index]
        np.clip(corrupted, -threshold, threshold, out=corrupted)

    elif corruption == "dropout":
        fraction = (0.10, 0.30, 0.50)[severity_index]
        span = max(1, int(round(fraction * length)))
        starts = rng.integers(0, length - span + 1, size=count)
        for sample_index, start in enumerate(starts):
            corrupted[sample_index, 0, start:start + span] = 0.0

    else:
        raise ValueError(f"Unknown corruption: {corruption}")
    return corrupted


def training_result_row(checkpoint, model_name, checkpoint_path):
    metrics = checkpoint.get("metrics", {})
    human_metrics = checkpoint.get("validation_metrics", {})
    qrs = human_metrics.get("qrs", {})
    morph = human_metrics.get("morph", {})
    return {
        "model": model_name,
        "training_stage": checkpoint.get("task", "ecg_sqa"),
        "best_epoch": checkpoint.get("epoch"),
        "val_loss": metrics.get("loss", human_metrics.get("loss")),
        "val_bce_qrs": metrics.get("bce_qrs"),
        "val_bce_morph": metrics.get("bce_morph"),
        "val_auroc_qrs": metrics.get("auroc_qrs", qrs.get("auroc")),
        "val_auroc_morph": metrics.get("auroc_morph", morph.get("auroc")),
        "val_auprc_qrs": metrics.get("auprc_qrs", qrs.get("auprc")),
        "val_auprc_morph": metrics.get("auprc_morph", morph.get("auprc")),
        "checkpoint": os.path.abspath(checkpoint_path),
    }


def summarize_natural(predictions):
    rows = []
    for model_name, values in predictions.items():
        for task_index, task in enumerate(TASKS):
            scores = values[:, task_index]
            row = {
                "model": model_name,
                "task": task,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "p25": float(np.percentile(scores, 25)),
                "median": float(np.median(scores)),
                "p75": float(np.percentile(scores, 75)),
            }
            for threshold in (0.5, 0.8, 0.9):
                row[f"accepted_{threshold:.1f}"] = float(np.mean(scores >= threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def model_agreement(predictions):
    names = list(predictions)
    rows = []
    matrices = {}
    for task_index, task in enumerate(TASKS):
        matrix = np.corrcoef(
            np.stack([predictions[name][:, task_index] for name in names])
        )
        matrices[task] = matrix
        for row_index, first in enumerate(names):
            for column_index, second in enumerate(names):
                rows.append({
                    "task": task,
                    "model_a": first,
                    "model_b": second,
                    "pearson_r": float(matrix[row_index, column_index]),
                })
    return names, matrices, pd.DataFrame(rows)


def artifact_associations(records, predictions):
    rows = []
    burden = records["artifact_burden"]
    for model_name, values in predictions.items():
        for task_index, task in enumerate(TASKS):
            score = pd.Series(values[:, task_index])
            rows.append({
                "model": model_name,
                "task": task,
                "spearman_artifact_burden": float(score.corr(burden, method="spearman")),
                "spearman_missing": float(
                    score.corr(records["missing_fraction"], method="spearman")
                ),
                "spearman_flat": float(
                    score.corr(records["flat_fraction"], method="spearman")
                ),
                "spearman_clipping": float(
                    score.corr(records["clipping_fraction"], method="spearman")
                ),
                "spearman_impulse": float(
                    score.corr(records["impulse_ratio"], method="spearman")
                ),
            })
    return pd.DataFrame(rows)


def run_stress_test(models, inputs, clean_predictions, batch_size, device, seed):
    rows = []
    for corruption_index, corruption in enumerate(CORRUPTIONS):
        for severity_index, severity in enumerate(SEVERITY_NAMES):
            corrupted = apply_corruption(
                inputs,
                corruption,
                severity_index,
                seed + 100 * corruption_index + severity_index,
            )
            for model_name, model in models.items():
                corrupted_scores = predict(model, corrupted, batch_size, device)
                clean_scores = clean_predictions[model_name]
                for task_index, task in enumerate(TASKS):
                    delta = corrupted_scores[:, task_index] - clean_scores[:, task_index]
                    rows.append({
                        "model": model_name,
                        "task": task,
                        "corruption": corruption,
                        "severity": severity,
                        "severity_index": severity_index,
                        "clean_mean": float(np.mean(clean_scores[:, task_index])),
                        "corrupted_mean": float(np.mean(corrupted_scores[:, task_index])),
                        "mean_delta": float(np.mean(delta)),
                        "median_delta": float(np.median(delta)),
                        "fraction_decreased": float(np.mean(delta < 0.0)),
                    })
    return pd.DataFrame(rows)


def plot_score_distributions(predictions, output_path):
    names = list(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 31)
    for axis, model_name in zip(axes.flat, names):
        values = predictions[model_name]
        axis.hist(values[:, 0], bins=bins, alpha=0.60, label="p_qrs")
        axis.hist(values[:, 1], bins=bins, alpha=0.60, label="p_morph")
        axis.set_title(model_name)
        axis.set_xlabel("Predicted probability")
        axis.set_ylabel("Windows")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("SQA score distributions on uncleaned ECG")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_acceptance_rates(summary, output_path):
    names = list(dict.fromkeys(summary["model"]))
    thresholds = (0.5, 0.8, 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(thresholds))
    width = 0.8 / len(names)

    for task_index, task in enumerate(TASKS):
        axis = axes[task_index]
        task_rows = summary[summary["task"] == task].set_index("model")
        for model_index, name in enumerate(names):
            rates = [task_rows.loc[name, f"accepted_{threshold:.1f}"] for threshold in thresholds]
            axis.bar(
                x - 0.4 + width / 2 + model_index * width,
                rates,
                width=width,
                label=name,
            )
        axis.set_xticks(x, [str(value) for value in thresholds])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{task.upper()} acceptance")
        axis.set_xlabel("Threshold")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accepted fraction")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_agreement(names, matrices, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, task in zip(axes, TASKS):
        matrix = matrices[task]
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.set_yticks(range(len(names)), names)
        axis.set_title(f"{task.upper()} model agreement")
        for row in range(len(names)):
            for column in range(len(names)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    fig.colorbar(image, ax=axes, shrink=0.75, label="Pearson r")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_artifact_strata(records, predictions, output_path):
    names = list(predictions)
    quantiles = pd.qcut(
        records["artifact_burden"],
        q=4,
        labels=False,
        duplicates="drop",
    )
    if quantiles.notna().sum() == 0:
        quantiles = pd.Series(np.zeros(len(records), dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for task_index, task in enumerate(TASKS):
        axis = axes[task_index]
        for name in names:
            scores = pd.Series(predictions[name][:, task_index])
            means = scores.groupby(quantiles).mean()
            axis.plot(
                means.index + 1,
                means.values,
                marker="o",
                label=name,
            )
        axis.set_title(f"{task.upper()} vs raw artifact burden")
        axis.set_xlabel("Artifact-burden quartile (higher = worse)")
        axis.set_xticks(range(1, int(quantiles.max()) + 2))
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean predicted probability")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stress(stress, task, output_path):
    names = list(dict.fromkeys(stress["model"]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    x = np.arange(3)
    for axis, model_name in zip(axes.flat, names):
        subset = stress[(stress["model"] == model_name) & (stress["task"] == task)]
        for corruption in CORRUPTIONS:
            rows = subset[subset["corruption"] == corruption].sort_values("severity_index")
            axis.plot(x, rows["mean_delta"], marker="o", label=corruption)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, SEVERITY_NAMES)
        axis.set_title(model_name)
        axis.set_ylabel("Mean probability change")
        axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(f"{task.upper()} sensitivity to controlled corruption")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_examples(inputs, records, predictions, output_path):
    ensemble = np.mean(
        np.stack([values.mean(axis=1) for values in predictions.values()]),
        axis=0,
    )
    order = np.argsort(ensemble)
    selected = [
        order[0],
        order[min(1, len(order) - 1)],
        order[len(order) // 2],
        order[min(len(order) // 2 + 1, len(order) - 1)],
        order[max(0, len(order) - 2)],
        order[-1],
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    names = list(predictions)
    for axis, index in zip(axes.flat, selected):
        axis.plot(inputs[index, 0], linewidth=0.8)
        model_summary = ", ".join(
            f"{name}={predictions[name][index].mean():.2f}" for name in names
        )
        axis.set_title(
            f"{records.loc[index, 'mirror']}/{records.loc[index, 'patient_id']} | "
            f"artifact={records.loc[index, 'artifact_burden']:.2f}\n{model_summary}",
            fontsize=8,
        )
        axis.grid(alpha=0.15)
    fig.suptitle("Examples ordered from low to high ensemble SQA score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_table(frame, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(path, training, natural, stress, window_count, file_count):
    natural_pivot = natural.pivot(index="model", columns="task")
    natural_rows = []
    for model_name in natural_pivot.index:
        natural_rows.append({
            "model": model_name,
            "mean_p_qrs": natural_pivot.loc[model_name, ("mean", "qrs")],
            "mean_p_morph": natural_pivot.loc[model_name, ("mean", "morph")],
            "qrs_accept_0.8": natural_pivot.loc[model_name, ("accepted_0.8", "qrs")],
            "morph_accept_0.8": natural_pivot.loc[model_name, ("accepted_0.8", "morph")],
        })
    natural_report = pd.DataFrame(natural_rows)

    severe = stress[stress["severity"] == "severe"]
    severe_delta = severe.pivot_table(
        index=["model", "task"],
        columns="corruption",
        values="mean_delta",
    ).reset_index()
    severe_decreased = severe.pivot_table(
        index=["model", "task"],
        columns="corruption",
        values="fraction_decreased",
    ).reset_index()

    with open(path, "w", encoding="utf-8") as file:
        file.write("# Unlabeled raw-ECG SQA evaluation\n\n")
        file.write(
            "**Interpretation:** the raw dataset has no human SQA labels. Natural-data "
            "accuracy/AUROC cannot be identified. The results below measure deployment "
            "behavior and controlled-corruption sensitivity, not clinical validity.\n\n"
        )
        file.write(f"Evaluated {window_count} windows from {file_count} patient files.\n\n")
        file.write("## Weak-label validation results\n\n")
        file.write(_markdown_table(training, [
            "model", "best_epoch", "val_loss", "val_auroc_qrs", "val_auroc_morph"
        ]))
        file.write("\n\n## Natural raw-data summary\n\n")
        file.write(_markdown_table(natural_report, [
            "model", "mean_p_qrs", "mean_p_morph", "qrs_accept_0.8", "morph_accept_0.8"
        ]))
        corruption_columns = [
            "model", "task", "gaussian", "baseline", "impulse", "clipping", "dropout"
        ]
        file.write("\n\n## Severe-corruption mean probability change\n\n")
        file.write(_markdown_table(severe_delta, corruption_columns))
        file.write("\n\n## Fraction of samples whose probability decreased\n\n")
        file.write(_markdown_table(severe_decreased, corruption_columns))
        file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SQA checkpoints on unlabeled mirror*_data ECG"
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows-per-file", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--stress-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.windows_per_file < 1:
        raise ValueError("--windows-per-file must be positive")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    output_dir = os.path.abspath(args.output_dir)
    if os.path.isdir(output_dir) and os.listdir(output_dir) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty evaluation directory: {output_dir}"
        )
    os.makedirs(output_dir, exist_ok=True)

    models = {}
    checkpoints = {}
    training_rows = []
    target_length = None
    window_sec = None

    for checkpoint_path in args.checkpoints:
        model, checkpoint = load_sqa_model(checkpoint_path, device)
        model_name = _model_identity(checkpoint)
        if model_name in models:
            raise ValueError(f"Duplicate model identity: {model_name}")
        models[model_name] = model
        checkpoints[model_name] = checkpoint
        training_rows.append(training_result_row(checkpoint, model_name, checkpoint_path))

        current_length = int(checkpoint["target_length"])
        current_window = float(checkpoint["window_sec"])
        target_length = current_length if target_length is None else target_length
        window_sec = current_window if window_sec is None else window_sec
        if current_length != target_length or current_window != window_sec:
            raise ValueError("All checkpoints must use the same target length and window duration.")

    inputs, records = load_raw_windows(
        args.data_root,
        window_sec,
        target_length,
        args.windows_per_file,
        args.max_files,
    )

    predictions = {}
    for model_name, model in models.items():
        print(f"[Inference] {model_name}...")
        predictions[model_name] = predict(model, inputs, args.batch_size, device)
        records[f"{model_name}_p_qrs"] = predictions[model_name][:, 0]
        records[f"{model_name}_p_morph"] = predictions[model_name][:, 1]

    training = pd.DataFrame(training_rows).sort_values("model")
    natural = summarize_natural(predictions)
    names, agreement_matrices, agreement = model_agreement(predictions)
    associations = artifact_associations(records, predictions)

    rng = np.random.default_rng(args.seed)
    stress_count = min(args.stress_samples, len(inputs))
    stress_indices = rng.choice(len(inputs), size=stress_count, replace=False)
    stress_inputs = inputs[stress_indices]
    clean_stress_predictions = {
        name: values[stress_indices] for name, values in predictions.items()
    }
    stress = run_stress_test(
        models,
        stress_inputs,
        clean_stress_predictions,
        args.batch_size,
        device,
        args.seed,
    )

    training.to_csv(os.path.join(output_dir, "training_results.csv"), index=False)
    natural.to_csv(os.path.join(output_dir, "natural_summary.csv"), index=False)
    agreement.to_csv(os.path.join(output_dir, "model_agreement.csv"), index=False)
    associations.to_csv(os.path.join(output_dir, "artifact_associations.csv"), index=False)
    stress.to_csv(os.path.join(output_dir, "perturbation_summary.csv"), index=False)
    records.to_csv(os.path.join(output_dir, "window_predictions.csv"), index=False)

    plot_score_distributions(
        predictions, os.path.join(output_dir, "score_distributions.png")
    )
    plot_acceptance_rates(
        natural, os.path.join(output_dir, "acceptance_rates.png")
    )
    plot_agreement(
        names, agreement_matrices, os.path.join(output_dir, "model_agreement.png")
    )
    plot_artifact_strata(
        records, predictions, os.path.join(output_dir, "artifact_strata.png")
    )
    for task in TASKS:
        plot_stress(
            stress,
            task,
            os.path.join(output_dir, f"perturbation_{task}.png"),
        )
    plot_examples(
        inputs,
        records,
        predictions,
        os.path.join(output_dir, "example_windows.png"),
    )

    file_count = int(records[["mirror", "patient_id"]].drop_duplicates().shape[0])
    write_report(
        os.path.join(output_dir, "evaluation_report.md"),
        training,
        natural,
        stress,
        len(records),
        file_count,
    )
    with open(os.path.join(output_dir, "evaluation_config.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print(f"[Done] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
