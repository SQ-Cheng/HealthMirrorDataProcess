"""Evaluate Exp3 checkpoints on full validation set with quantitative plots."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp3_dataloader import build_masked_recon_dataloaders
from exp3_model import build_exp3_model
from exp3_train import build_visible_mask, weighted_masked_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3 full-validation evaluation and visualization")
    parser.add_argument("--variant", type=str, default="light")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-block-min", type=int, default=8)
    parser.add_argument("--mask-block-max", type=int, default=32)
    parser.add_argument("--context-weight", type=float, default=0.20)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    return parser.parse_args()


def find_checkpoint(variant, explicit_path=None):
    if explicit_path:
        return explicit_path

    candidates = []
    if variant == "full":
        # Prefer current final direction first.
        candidates.extend(
            [
                os.path.join(CHECKPOINT_DIR, "exp3_full_ecgfocus_best.pt"),
                os.path.join(CHECKPOINT_DIR, "exp3_full_ecgfocus_final.pt"),
            ]
        )

    candidates.extend(
        [
            os.path.join(CHECKPOINT_DIR, f"exp3_{variant}_best.pt"),
            os.path.join(CHECKPOINT_DIR, f"exp3_{variant}_final.pt"),
        ]
    )

    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt

    tried = ", ".join(candidates)
    raise FileNotFoundError(f"No checkpoint found for {variant}. Tried: {tried}")


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def channel_metrics(pred, target, masked):
    p = pred[masked]
    t = target[masked]
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = safe_corr(t, p)
    return mae, rmse, corr


def channel_grad_mae(pred, target, masked):
    pred_d = pred[:, 1:] - pred[:, :-1]
    true_d = target[:, 1:] - target[:, :-1]
    mask_d = masked[:, 1:] & masked[:, :-1]
    if not np.any(mask_d):
        return 0.0
    return float(np.mean(np.abs(pred_d[mask_d] - true_d[mask_d])))


def evaluate(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = find_checkpoint(args.variant, args.checkpoint)

    model = build_exp3_model(args.variant).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = build_masked_recon_dataloaders(
        ROOT_DIR,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    criterion = nn.SmoothL1Loss(reduction="none")

    losses_model = []
    losses_zero = []
    all_pred_model = []
    all_pred_zero = []
    all_target = []
    all_masked = []
    all_visible = []
    all_input = []
    sample_rows = []

    sample_idx = 0
    with torch.no_grad():
        for pair, clean_score, _, _ in val_loader:
            pair = pair.to(DEVICE)
            clean_score = clean_score.to(DEVICE)

            visible = build_visible_mask(
                pair,
                mask_ratio=args.mask_ratio,
                block_min=args.mask_block_min,
                block_max=args.mask_block_max,
            )
            masked = 1.0 - visible
            x_masked = pair * visible

            pred_model = model(x_masked, visible)
            pred_zero = x_masked.clone()

            loss_model = weighted_masked_loss(
                pred_model,
                pair,
                visible,
                clean_score,
                criterion,
                context_weight=args.context_weight,
            )
            loss_zero = weighted_masked_loss(
                pred_zero,
                pair,
                visible,
                clean_score,
                criterion,
                context_weight=args.context_weight,
            )
            losses_model.append(float(loss_model.item()))
            losses_zero.append(float(loss_zero.item()))

            p_model = pred_model.detach().cpu().numpy()
            p_zero = pred_zero.detach().cpu().numpy()
            t = pair.detach().cpu().numpy()
            m = masked.detach().cpu().numpy() > 0.5
            v = visible.detach().cpu().numpy()
            x = x_masked.detach().cpu().numpy()
            c = clean_score.detach().cpu().numpy()

            all_pred_model.append(p_model)
            all_pred_zero.append(p_zero)
            all_target.append(t)
            all_masked.append(m)
            all_visible.append(v)
            all_input.append(x)

            sample_err_model = (np.abs(p_model - t) * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)), 1.0)
            sample_err_zero = (np.abs(p_zero - t) * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)), 1.0)

            for i in range(len(c)):
                sample_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "clean_score": float(c[i]),
                        "model_mae_masked": float(sample_err_model[i]),
                        "zero_mae_masked": float(sample_err_zero[i]),
                    }
                )
                sample_idx += 1

    pred_model = np.concatenate(all_pred_model, axis=0)
    pred_zero = np.concatenate(all_pred_zero, axis=0)
    target = np.concatenate(all_target, axis=0)
    masked = np.concatenate(all_masked, axis=0)
    visible = np.concatenate(all_visible, axis=0)
    x_masked = np.concatenate(all_input, axis=0)
    sample_df = pd.DataFrame(sample_rows)

    ecg_model = channel_metrics(pred_model[:, 0], target[:, 0], masked[:, 0])
    ecg_zero = channel_metrics(pred_zero[:, 0], target[:, 0], masked[:, 0])
    rppg_model = channel_metrics(pred_model[:, 1], target[:, 1], masked[:, 1])
    rppg_zero = channel_metrics(pred_zero[:, 1], target[:, 1], masked[:, 1])

    ecg_grad_model = channel_grad_mae(pred_model[:, 0], target[:, 0], masked[:, 0])
    ecg_grad_zero = channel_grad_mae(pred_zero[:, 0], target[:, 0], masked[:, 0])
    rppg_grad_model = channel_grad_mae(pred_model[:, 1], target[:, 1], masked[:, 1])
    rppg_grad_zero = channel_grad_mae(pred_zero[:, 1], target[:, 1], masked[:, 1])

    clean = sample_df["clean_score"].values
    err_model = sample_df["model_mae_masked"].values
    err_zero = sample_df["zero_mae_masked"].values

    q = np.quantile(clean, [0.25, 0.50, 0.75])
    edges = [-1e9, q[0], q[1], q[2], 1e9]
    q_model = []
    q_zero = []
    for i in range(4):
        mask_q = (clean >= edges[i]) & (clean < edges[i + 1])
        q_model.append(float(np.mean(err_model[mask_q])) if np.any(mask_q) else float("nan"))
        q_zero.append(float(np.mean(err_zero[mask_q])) if np.any(mask_q) else float("nan"))

    metrics = {
        "variant": args.variant,
        "checkpoint": ckpt_path,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "val_batches": int(len(losses_model)),
        "val_samples": int(len(sample_df)),
        "mask_ratio": float(args.mask_ratio),
        "weighted_loss_model": float(np.mean(losses_model)),
        "weighted_loss_zero": float(np.mean(losses_zero)),
        "ecg": {
            "mae_model": ecg_model[0],
            "mae_zero": ecg_zero[0],
            "rmse_model": ecg_model[1],
            "rmse_zero": ecg_zero[1],
            "corr_model": ecg_model[2],
            "corr_zero": ecg_zero[2],
            "grad_mae_model": ecg_grad_model,
            "grad_mae_zero": ecg_grad_zero,
        },
        "rppg": {
            "mae_model": rppg_model[0],
            "mae_zero": rppg_zero[0],
            "rmse_model": rppg_model[1],
            "rmse_zero": rppg_zero[1],
            "corr_model": rppg_model[2],
            "corr_zero": rppg_zero[2],
            "grad_mae_model": rppg_grad_model,
            "grad_mae_zero": rppg_grad_zero,
        },
        "corr_clean_vs_model_err": safe_corr(clean, err_model),
        "corr_clean_vs_zero_err": safe_corr(clean, err_zero),
        "quartile_model_mae_q1_to_q4": q_model,
        "quartile_zero_mae_q1_to_q4": q_zero,
    }

    return metrics, sample_df, target, x_masked, pred_model, masked, visible


def save_summary_plot(metrics, sample_df, out_path):
    clean = sample_df["clean_score"].values
    err_model = sample_df["model_mae_masked"].values
    err_zero = sample_df["zero_mae_masked"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: MAE/RMSE bars
    ax = axes[0, 0]
    labels = ["ECG", "rPPG"]
    x = np.arange(len(labels))
    w = 0.18
    ax.bar(x - 1.5 * w, [metrics["ecg"]["mae_model"], metrics["rppg"]["mae_model"]], w, label="Model MAE")
    ax.bar(x - 0.5 * w, [metrics["ecg"]["mae_zero"], metrics["rppg"]["mae_zero"]], w, label="Zero MAE")
    ax.bar(x + 0.5 * w, [metrics["ecg"]["rmse_model"], metrics["rppg"]["rmse_model"]], w, label="Model RMSE")
    ax.bar(x + 1.5 * w, [metrics["ecg"]["rmse_zero"], metrics["rppg"]["rmse_zero"]], w, label="Zero RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Masked Reconstruction Error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    # B: gradient MAE bars
    ax = axes[0, 1]
    labels_grad = ["ECG grad", "rPPG grad"]
    xg = np.arange(len(labels_grad))
    ax.bar(xg - 0.18, [metrics["ecg"]["grad_mae_model"], metrics["rppg"]["grad_mae_model"]], 0.36, label="Model")
    ax.bar(xg + 0.18, [metrics["ecg"]["grad_mae_zero"], metrics["rppg"]["grad_mae_zero"]], 0.36, label="Zero baseline")
    ax.set_xticks(xg)
    ax.set_xticklabels(labels_grad)
    ax.set_title("Gradient MAE (masked)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    # C: clean score vs model error
    ax = axes[1, 0]
    ax.scatter(clean, err_model, s=8, alpha=0.35, label="Model")
    ax.scatter(clean, err_zero, s=8, alpha=0.20, label="Zero baseline")
    if len(clean) > 1:
        px = np.linspace(clean.min(), clean.max(), 100)
        c1 = np.polyfit(clean, err_model, 1)
        c0 = np.polyfit(clean, err_zero, 1)
        ax.plot(px, c1[0] * px + c1[1], linewidth=1.5)
        ax.plot(px, c0[0] * px + c0[1], linewidth=1.5)
    ax.set_xlabel("clean_score")
    ax.set_ylabel("sample masked MAE")
    ax.set_title("Quality vs Error")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    # D: sample error histogram + corr text
    ax = axes[1, 1]
    ax.hist(err_zero, bins=40, alpha=0.5, label="Zero baseline")
    ax.hist(err_model, bins=40, alpha=0.6, label="Model")
    ax.axvline(float(np.mean(err_zero)), linestyle="--", linewidth=1)
    ax.axvline(float(np.mean(err_model)), linestyle="--", linewidth=1)
    ax.set_xlabel("sample masked MAE")
    ax.set_ylabel("count")
    ax.set_title("Validation Error Distribution")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    text = (
        f"corr ECG model={metrics['ecg']['corr_model']:.3f}, zero={metrics['ecg']['corr_zero']:.3f}\n"
        f"corr rPPG model={metrics['rppg']['corr_model']:.3f}, zero={metrics['rppg']['corr_zero']:.3f}\n"
        f"corr(clean,err) model={metrics['corr_clean_vs_model_err']:.3f}, zero={metrics['corr_clean_vs_zero_err']:.3f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.7})

    fig.suptitle(
        "Exp3 Validation Performance Summary\n"
        f"variant={metrics['variant']} | best_epoch={metrics['best_epoch']} | "
        f"weighted_loss model={metrics['weighted_loss_model']:.4f}, zero={metrics['weighted_loss_zero']:.4f}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)


def save_corr_plot(metrics, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.2))
    corr_names = ["ECG corr", "rPPG corr", "clean vs err"]
    model_corr = [
        metrics["ecg"]["corr_model"],
        metrics["rppg"]["corr_model"],
        metrics["corr_clean_vs_model_err"],
    ]
    zero_corr = [
        metrics["ecg"]["corr_zero"],
        metrics["rppg"]["corr_zero"],
        metrics["corr_clean_vs_zero_err"],
    ]
    xc = np.arange(len(corr_names))
    ax.bar(xc - 0.18, model_corr, 0.36, label="Model")
    ax.bar(xc + 0.18, zero_corr, 0.36, label="Zero baseline")
    ax.set_xticks(xc)
    ax.set_xticklabels(corr_names, rotation=10)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Correlation Metrics")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)


def save_segment_plot(args, sample_df, target, x_masked, pred_model, masked, out_path):
    n = min(max(1, args.num_segments), len(sample_df))
    order = np.argsort(sample_df["model_mae_masked"].values)
    picks = np.linspace(0, len(order) - 1, n).astype(int)
    idx = order[picks]

    fs_target = args.target_length / args.window_sec
    t = np.arange(args.target_length) / fs_target

    fig, axes = plt.subplots(n, 2, figsize=(14, max(3 * n, 4)), sharex=True)
    if n == 1:
        axes = np.array([axes])

    for i, sid in enumerate(idx):
        ax0 = axes[i, 0]
        ax1 = axes[i, 1]

        ax0.plot(t, target[sid, 0], color="tab:blue", linewidth=1.0, label="target")
        ax0.plot(t, x_masked[sid, 0], color="tab:gray", linewidth=0.9, alpha=0.7, label="masked input")
        ax0.plot(t, pred_model[sid, 0], color="tab:orange", linewidth=1.0, label="model")
        ax0.set_title(
            f"ECG sid={sid} | clean={sample_df.iloc[sid]['clean_score']:.3f} | "
            f"maskedMAE={sample_df.iloc[sid]['model_mae_masked']:.3f}"
        )
        ax0.grid(alpha=0.2)

        ax1.plot(t, target[sid, 1], color="tab:green", linewidth=1.0, label="target")
        ax1.plot(t, x_masked[sid, 1], color="tab:gray", linewidth=0.9, alpha=0.7, label="masked input")
        ax1.plot(t, pred_model[sid, 1], color="tab:red", linewidth=1.0, label="model")
        ax1.set_title(
            f"rPPG sid={sid} | mask_frac={float(masked[sid, 1].mean()):.2f}"
        )
        ax1.grid(alpha=0.2)

        for ax in (ax0, ax1):
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized amplitude")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)

    fig.suptitle("Exp3 Representative Validation Segments (low to high error)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)


def main():
    args = parse_args()

    if args.output_dir is None:
        out_dir = os.path.join(ROOT_DIR, "train", "exp3", "plots")
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    metrics, sample_df, target, x_masked, pred_model, masked, _ = evaluate(args)

    tag = args.output_tag if args.output_tag else args.variant
    summary_plot = os.path.join(out_dir, f"exp3_eval_{tag}_summary.png")
    corr_plot = os.path.join(out_dir, f"exp3_eval_{tag}_corr.png")
    seg_plot = os.path.join(out_dir, f"exp3_eval_{tag}_segments.png")
    csv_path = os.path.join(out_dir, f"exp3_eval_{tag}_samples.csv")
    json_path = os.path.join(out_dir, f"exp3_eval_{tag}_metrics.json")

    save_summary_plot(metrics, sample_df, summary_plot)
    save_corr_plot(metrics, corr_plot)
    save_segment_plot(args, sample_df, target, x_masked, pred_model, masked, seg_plot)
    sample_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {summary_plot}")
    print(f"Saved: {corr_plot}")
    print(f"Saved: {seg_plot}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")

    print("\n[Exp3 Eval]", args.variant)
    print(
        f"weighted_loss model={metrics['weighted_loss_model']:.4f} | "
        f"zero={metrics['weighted_loss_zero']:.4f}"
    )
    print(
        f"ECG MAE model={metrics['ecg']['mae_model']:.4f}, zero={metrics['ecg']['mae_zero']:.4f} | "
        f"corr model={metrics['ecg']['corr_model']:.4f}, zero={metrics['ecg']['corr_zero']:.4f}"
    )
    print(
        f"rPPG MAE model={metrics['rppg']['mae_model']:.4f}, zero={metrics['rppg']['mae_zero']:.4f} | "
        f"corr model={metrics['rppg']['corr_model']:.4f}, zero={metrics['rppg']['corr_zero']:.4f}"
    )
    print(
        f"corr(clean_score, err) model={metrics['corr_clean_vs_model_err']:.4f}, "
        f"zero={metrics['corr_clean_vs_zero_err']:.4f}"
    )


if __name__ == "__main__":
    main()
