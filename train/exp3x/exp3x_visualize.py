"""Experiment 03X evaluation and visualization for trained models."""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
EXP3_DIR = os.path.join(ROOT_DIR, "train", "exp3")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "train", "checkpoints")
PLOT_DIR = os.path.join(ROOT_DIR, "train", "exp3x", "plots")

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, EXP3_DIR)

from exp3x_model import build_exp3x_model

try:
    from train.exp3.exp3_dataloader import build_masked_recon_dataloaders
    from train.exp3.exp3_train import build_visible_mask, weighted_masked_loss
except ModuleNotFoundError:
    from exp3_dataloader import build_masked_recon_dataloaders  # type: ignore[import-not-found]
    from exp3_train import build_visible_mask, weighted_masked_loss  # type: ignore[import-not-found]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3X full-validation evaluation and visualization")
    parser.add_argument("--model", choices=["unet_gated", "dual_head", "tcn_ssm", "cross_attention"], default="tcn_ssm")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional data root directory containing mirror*_auto_cleaned[_sqi] folders",
    )
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
    parser.add_argument("--ecg-point-weight", type=float, default=1.25)
    parser.add_argument("--rppg-point-weight", type=float, default=1.0)
    parser.add_argument("--grad-loss-weight", type=float, default=0.1)
    parser.add_argument("--ecg-fft-loss-weight", type=float, default=0.02)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--output-tag", type=str, default="")
    return parser.parse_args()


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


def find_checkpoint(model_name, explicit_path=None):
    if explicit_path:
        return explicit_path

    candidates = [
        os.path.join(CHECKPOINT_DIR, f"exp3x_{model_name}_best.pt"),
        os.path.join(CHECKPOINT_DIR, f"exp3x_{model_name}_final.pt"),
    ]

    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(f"No checkpoint found for {model_name}. Tried: {', '.join(candidates)}")


def evaluate(args):
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else ROOT_DIR
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_path = find_checkpoint(args.model, args.checkpoint)

    model = build_exp3x_model(args.model).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader = build_masked_recon_dataloaders(
        data_dir,
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
    all_pred = []
    all_zero = []
    all_target = []
    all_masked = []
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

            pred = model(x_masked, visible)
            zero = x_masked.clone()

            loss_model = weighted_masked_loss(
                pred,
                pair,
                visible,
                clean_score,
                criterion,
                context_weight=args.context_weight,
                ecg_point_weight=args.ecg_point_weight,
                rppg_point_weight=args.rppg_point_weight,
                grad_loss_weight=args.grad_loss_weight,
                ecg_fft_loss_weight=args.ecg_fft_loss_weight,
            )
            loss_zero = weighted_masked_loss(
                zero,
                pair,
                visible,
                clean_score,
                criterion,
                context_weight=args.context_weight,
                ecg_point_weight=args.ecg_point_weight,
                rppg_point_weight=args.rppg_point_weight,
                grad_loss_weight=args.grad_loss_weight,
                ecg_fft_loss_weight=args.ecg_fft_loss_weight,
            )
            losses_model.append(float(loss_model.item()))
            losses_zero.append(float(loss_zero.item()))

            p = pred.detach().cpu().numpy()
            z = zero.detach().cpu().numpy()
            t = pair.detach().cpu().numpy()
            m = masked.detach().cpu().numpy() > 0.5
            x = x_masked.detach().cpu().numpy()
            c = clean_score.detach().cpu().numpy()

            all_pred.append(p)
            all_zero.append(z)
            all_target.append(t)
            all_masked.append(m)
            all_input.append(x)

            err_model = (np.abs(p - t) * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)), 1.0)
            err_zero = (np.abs(z - t) * m).sum(axis=(1, 2)) / np.maximum(m.sum(axis=(1, 2)), 1.0)
            for i in range(len(c)):
                sample_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "clean_score": float(c[i]),
                        "model_mae_masked": float(err_model[i]),
                        "zero_mae_masked": float(err_zero[i]),
                    }
                )
                sample_idx += 1

    pred = np.concatenate(all_pred, axis=0)
    zero = np.concatenate(all_zero, axis=0)
    target = np.concatenate(all_target, axis=0)
    masked = np.concatenate(all_masked, axis=0)
    x_masked = np.concatenate(all_input, axis=0)
    sample_df = pd.DataFrame(sample_rows)

    ecg_model = channel_metrics(pred[:, 0], target[:, 0], masked[:, 0])
    ecg_zero = channel_metrics(zero[:, 0], target[:, 0], masked[:, 0])
    rppg_model = channel_metrics(pred[:, 1], target[:, 1], masked[:, 1])
    rppg_zero = channel_metrics(zero[:, 1], target[:, 1], masked[:, 1])

    metrics = {
        "model": args.model,
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
            "grad_mae_model": channel_grad_mae(pred[:, 0], target[:, 0], masked[:, 0]),
            "grad_mae_zero": channel_grad_mae(zero[:, 0], target[:, 0], masked[:, 0]),
        },
        "rppg": {
            "mae_model": rppg_model[0],
            "mae_zero": rppg_zero[0],
            "rmse_model": rppg_model[1],
            "rmse_zero": rppg_zero[1],
            "corr_model": rppg_model[2],
            "corr_zero": rppg_zero[2],
            "grad_mae_model": channel_grad_mae(pred[:, 1], target[:, 1], masked[:, 1]),
            "grad_mae_zero": channel_grad_mae(zero[:, 1], target[:, 1], masked[:, 1]),
        },
        "corr_clean_vs_model_err": safe_corr(sample_df["clean_score"].values, sample_df["model_mae_masked"].values),
        "corr_clean_vs_zero_err": safe_corr(sample_df["clean_score"].values, sample_df["zero_mae_masked"].values),
    }

    return metrics, sample_df, target, x_masked, pred, masked


def save_summary_plot(metrics, sample_df, out_path):
    clean = sample_df["clean_score"].values
    err_model = sample_df["model_mae_masked"].values
    err_zero = sample_df["zero_mae_masked"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    labels = ["ECG", "rPPG"]
    x = np.arange(len(labels))
    w = 0.18
    axes[0, 0].bar(x - 1.5 * w, [metrics["ecg"]["mae_model"], metrics["rppg"]["mae_model"]], w, label="Model MAE")
    axes[0, 0].bar(x - 0.5 * w, [metrics["ecg"]["mae_zero"], metrics["rppg"]["mae_zero"]], w, label="Zero MAE")
    axes[0, 0].bar(x + 0.5 * w, [metrics["ecg"]["rmse_model"], metrics["rppg"]["rmse_model"]], w, label="Model RMSE")
    axes[0, 0].bar(x + 1.5 * w, [metrics["ecg"]["rmse_zero"], metrics["rppg"]["rmse_zero"]], w, label="Zero RMSE")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_title("Masked Reconstruction Error")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    labels_grad = ["ECG grad", "rPPG grad"]
    xg = np.arange(len(labels_grad))
    axes[0, 1].bar(xg - 0.18, [metrics["ecg"]["grad_mae_model"], metrics["rppg"]["grad_mae_model"]], 0.36, label="Model")
    axes[0, 1].bar(xg + 0.18, [metrics["ecg"]["grad_mae_zero"], metrics["rppg"]["grad_mae_zero"]], 0.36, label="Zero")
    axes[0, 1].set_xticks(xg)
    axes[0, 1].set_xticklabels(labels_grad)
    axes[0, 1].set_title("Gradient MAE (masked)")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].scatter(clean, err_model, s=8, alpha=0.35, label="Model")
    axes[1, 0].scatter(clean, err_zero, s=8, alpha=0.20, label="Zero")
    axes[1, 0].set_xlabel("clean_score")
    axes[1, 0].set_ylabel("sample masked MAE")
    axes[1, 0].set_title("Quality vs Error")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(err_zero, bins=40, alpha=0.5, label="Zero")
    axes[1, 1].hist(err_model, bins=40, alpha=0.6, label="Model")
    axes[1, 1].set_xlabel("sample masked MAE")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("Validation Error Distribution")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    text = (
        f"ECG corr model={metrics['ecg']['corr_model']:.3f}, zero={metrics['ecg']['corr_zero']:.3f}\\n"
        f"rPPG corr model={metrics['rppg']['corr_model']:.3f}, zero={metrics['rppg']['corr_zero']:.3f}\\n"
        f"corr(clean,err) model={metrics['corr_clean_vs_model_err']:.3f}, zero={metrics['corr_clean_vs_zero_err']:.3f}"
    )
    axes[1, 1].text(0.02, 0.98, text, transform=axes[1, 1].transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75})

    fig.suptitle(
        f"Exp3X Validation Summary | {metrics['model']} | epoch={metrics['best_epoch']} | "
        f"weighted model={metrics['weighted_loss_model']:.4f}, zero={metrics['weighted_loss_zero']:.4f}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)


def save_segments_plot(args, sample_df, target, x_masked, pred, masked, out_path):
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
        ax0.plot(t, pred[sid, 0], color="tab:orange", linewidth=1.0, label="model")
        ym = masked[sid, 0]
        if np.any(ym):
            y0_min = min(target[sid, 0].min(), pred[sid, 0].min())
            y0_max = max(target[sid, 0].max(), pred[sid, 0].max())
            ax0.fill_between(t, y0_min, y0_max, where=ym, color="tab:red", alpha=0.10)
        ax0.set_title(f"ECG sample={sid} mae={sample_df.iloc[sid]['model_mae_masked']:.3f}")
        ax0.grid(alpha=0.2)

        ax1.plot(t, target[sid, 1], color="tab:green", linewidth=1.0, label="target")
        ax1.plot(t, x_masked[sid, 1], color="tab:gray", linewidth=0.9, alpha=0.7, label="masked input")
        ax1.plot(t, pred[sid, 1], color="tab:orange", linewidth=1.0, label="model")
        ym = masked[sid, 1]
        if np.any(ym):
            y1_min = min(target[sid, 1].min(), pred[sid, 1].min())
            y1_max = max(target[sid, 1].max(), pred[sid, 1].max())
            ax1.fill_between(t, y1_min, y1_max, where=ym, color="tab:red", alpha=0.10)
        ax1.set_title(f"rPPG sample={sid} mae={sample_df.iloc[sid]['model_mae_masked']:.3f}")
        ax1.grid(alpha=0.2)

        if i == 0:
            ax0.legend(fontsize=8, loc="upper right")
            ax1.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Exp3X reconstruction segments ({args.model})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)


def main():
    args = parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)

    tag = args.output_tag.strip() if args.output_tag.strip() else args.model
    metrics_path = os.path.join(PLOT_DIR, f"exp3x_eval_{tag}_metrics.json")
    samples_path = os.path.join(PLOT_DIR, f"exp3x_eval_{tag}_samples.csv")
    summary_png = os.path.join(PLOT_DIR, f"exp3x_eval_{tag}_summary.png")
    segments_png = os.path.join(PLOT_DIR, f"exp3x_eval_{tag}_segments.png")

    metrics, sample_df, target, x_masked, pred, masked = evaluate(args)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    sample_df.to_csv(samples_path, index=False)

    save_summary_plot(metrics, sample_df, summary_png)
    save_segments_plot(args, sample_df, target, x_masked, pred, masked, segments_png)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {samples_path}")
    print(f"Saved: {summary_png}")
    print(f"Saved: {segments_png}")
    print(
        f"WeightedLoss={metrics['weighted_loss_model']:.6f} | "
        f"ECG_MAE={metrics['ecg']['mae_model']:.6f} | "
        f"rPPG_MAE={metrics['rppg']['mae_model']:.6f}"
    )


if __name__ == "__main__":
    main()
