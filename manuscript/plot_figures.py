"""Render manuscript figures directly from completed experiment CSVs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

TARGETS = [
    ("hemoglobin_low", "Hemoglobin"),
    ("lactate_high", "Lactate"),
    ("oxyhemoglobin_fraction", "Oxyhemoglobin"),
    ("aa_po2_ratio_low", "A/a PO2 ratio"),
    ("total_bilirubin_high", "Bilirubin"),
    ("creatinine_high", "Creatinine"),
    ("urea_high", "Urea"),
    ("platelet_count_low", "Platelets"),
]
PAIR_TARGETS = TARGETS[:5] + [("troponin_high", "Troponin")] + TARGETS[5:]
COLORS = {"face": "#177E89", "diverse": "#D38937", "video": "#56616E",
          "history": "#35739D", "combined": "#BF6472"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42,
    })


def cv(path, metric):
    data = pd.read_csv(ROOT / path)
    return data.loc[data.metric.eq(metric)].set_index("target")


def holdout(path, metric):
    data = pd.read_csv(ROOT / path)
    return data.loc[data.split.eq("test")].set_index("target")[metric]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.11)
    plt.close(fig)


def cabg_trajectories():
    source = (ROOT / "study/exp2_lab_longitudinal_statistics/outputs/tables"
              / "surgery_phase_summary.csv")
    data = pd.read_csv(source)
    data = data.loc[data.cohort.eq("cabg")].set_index(["variable_id", "surgery_phase"])
    phases = ("pre_3_7d", "pre_1_3d", "post_0_6h", "post_6_24h",
              "post_1_2d", "post_2_3d", "post_3_7d")
    labels = ("Pre\n3-7 d", "Pre\n1-3 d", "Post\n0-6 h", "Post\n6-24 h",
              "Post\n1-2 d", "Post\n2-3 d", "Post\n3-7 d")
    analytes = (
        ("A021", "Lactate", "mmol/L", "#187E89"),
        ("A102", "Hemoglobin", "g/L", "#B86443"),
        ("A080", "Platelet count", r"$10^9$/L", "#55749A"),
        ("A074", "Creatinine", r"$\mu$mol/L", "#6B7257"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.45), sharex=True)
    for ax, (key, name, unit, color) in zip(axes.flat, analytes):
        for segment in (range(2), range(2, len(phases))):
            points = [(i, data.loc[(key, phases[i])]) for i in segment
                      if (key, phases[i]) in data.index
                      and data.loc[(key, phases[i]), "patients"] >= 20]
            if not points:
                continue
            x = np.array([i for i, _ in points])
            med = np.array([row["median"] for _, row in points], dtype=float)
            low = np.array([row["q25"] for _, row in points], dtype=float)
            high = np.array([row["q75"] for _, row in points], dtype=float)
            ax.errorbar(x, med, yerr=[med - low, high - med], fmt="o-",
                        color=color, linewidth=1.4, elinewidth=0.9,
                        capsize=2.3, markersize=3.3, zorder=3)
            for position, top, (_, row) in zip(x, high, points):
                ax.annotate(str(int(row["patients"])), (position, top),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.2,
                            color="#4F5960")
        ax.axvline(1.5, color="#9AA2A7", linestyle="--", linewidth=0.8)
        ax.set_title(name, loc="left", fontweight="semibold")
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(phases)), labels, fontsize=7.1)
        ax.grid(axis="y", color="#E5E8E9", linewidth=0.7)
        values = data.loc[key].reindex(phases)
        values = values.loc[values.patients >= 20]
        span = values.q75.max() - values.q25.min()
        ax.set_ylim(values.q25.min() - 0.08 * span,
                    values.q75.max() + 0.26 * span)
    for ax in axes.flat:
        ax.tick_params(axis="x", labelbottom=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.12,
                        hspace=0.35, wspace=0.28)
    save(fig, "cabg_selected_trajectories.pdf")


def regression_cv():
    paths = {
        "Face-only EfficientNet-B0": "study/exp2_face_pretrained_head32_regression/outputs/5fold/cv_summary.csv",
        "Patient-diverse EfficientNet-B0": "study/exp2_face_pretrained_head32_regression/outputs/ablations/patient_diverse_schedule_30_40/5fold/cv_summary.csv",
        "R3D-18 video": "study/exp3_video_lab_regression/outputs/5fold/cv_summary.csv",
    }
    colors = [COLORS["face"], COLORS["diverse"], COLORS["video"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    y = np.arange(len(TARGETS))
    offsets = [-0.23, 0, 0.23]
    for (label, path), color, offset in zip(paths.items(), colors, offsets):
        data = cv(path, "r2")
        values = [data.loc[key, "pooled_oof"] for key, _ in TARGETS]
        ax.barh(y + offset, values, height=0.20, color=color, label=label, zorder=3)
    counts = cv(next(iter(paths.values())), "r2")["videos"]
    ax.set_yticks(y, [f"{name}   n={int(counts.loc[key]):,}"
                      for key, name in TARGETS])
    ax.invert_yaxis()
    ax.axvline(0, color="#9CA3A8", lw=0.8)
    ax.set_xlim(-0.08, 0.32)
    ax.set_xlabel(r"Pooled out-of-fold video-level $R^2$")
    ax.legend(loc="lower center", frameon=False, ncol=2,
              bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.grid(axis="x", color="#E4E7E9", lw=0.7)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.77, bottom=0.15)
    save(fig, "exp2_exp3_cv_regression.pdf")


def classification_cv():
    path = "study/exp2_face_pretrained_head32_classification/outputs/5fold/cv_summary.csv"
    auc = cv(path, "roc_auc")
    bacc = cv(path, "balanced_accuracy")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(TARGETS))
    ax.barh(y - 0.16, [auc.loc[key, "pooled_oof"] for key, _ in TARGETS],
            height=0.29, color=COLORS["face"], label="AUROC", zorder=3)
    ax.barh(y + 0.16, [bacc.loc[key, "pooled_oof"] for key, _ in TARGETS],
            height=0.29, color=COLORS["diverse"], label="Balanced accuracy", zorder=3)
    ax.set_yticks(y, [f"{name}   n={int(auc.loc[key, 'videos']):,}"
                      for key, name in TARGETS])
    ax.invert_yaxis()
    ax.axvline(0.5, color="#9CA3A8", lw=0.8)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Pooled out-of-fold video-level score")
    ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.0))
    ax.grid(axis="x", color="#E4E7E9", lw=0.7)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.84, bottom=0.15)
    save(fig, "exp2_cv_classification.pdf")


def history_comparison():
    paths = {
        "Face-only": "study/exp2_face_pretrained_head32_regression/outputs/20frame/metrics_all.csv",
        "History-only": "study/exp2_history_only_head32_regression/outputs/metrics_all.csv",
        "Face + history": "study/exp2_face_history_head32_regression/outputs/20frame/metrics_all.csv",
    }
    colors = [COLORS["face"], COLORS["history"], COLORS["combined"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    y = np.arange(len(TARGETS))
    for (label, path), color, offset in zip(paths.items(), colors, [-0.22, 0, 0.22]):
        values = holdout(path, "r2")
        ax.barh(y + offset, [values.loc[key] for key, _ in TARGETS],
                height=0.20, color=color, label=label, zorder=3)
    counts = holdout(next(iter(paths.values())), "n")
    ax.set_yticks(y, [f"{name}   n={int(counts.loc[key]):,}"
                      for key, name in TARGETS])
    ax.invert_yaxis()
    ax.axvline(0, color="#9CA3A8", lw=0.8)
    ax.set_xlim(-0.21, 0.86)
    ax.set_xlabel(r"Original held-out test video-level $R^2$")
    ax.legend(loc="lower right", frameon=False, ncol=3, bbox_to_anchor=(1, 1.0))
    ax.grid(axis="x", color="#E4E7E9", lw=0.7)
    fig.subplots_adjust(left=0.33, right=0.97, top=0.84, bottom=0.15)
    save(fig, "exp2_history_ablation.pdf")


def pair_delta():
    reg = pd.read_csv(ROOT / "study/exp6_face_pair_lab_delta/outputs/metrics_all.csv")
    cls = pd.read_csv(ROOT / "study/exp6_face_pair_lab_delta_classifacation/outputs/metrics_all.csv")
    reg = reg.loc[reg.split.eq("test")].set_index("target")
    cls = cls.loc[cls.split.eq("test")].set_index("target")
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.4),
                             gridspec_kw={"hspace": 0.38})
    y = np.arange(len(PAIR_TARGETS))
    ax = axes[0]
    ax.barh(y, [reg.loc[key, "r2"] for key, _ in PAIR_TARGETS],
            height=0.62, color=COLORS["face"], zorder=3)
    ax.axvline(0, color="#9CA3A8", lw=0.8)
    ax.set_xlim(-0.21, 0.43)
    ax.set_yticks(y, [f"{name}   n={int(reg.loc[key, 'n']):,}"
                      for key, name in PAIR_TARGETS])
    ax.invert_yaxis()
    ax.set_xlabel(r"Delta regression $R^2$")
    ax.grid(axis="x", color="#E4E7E9", lw=0.7)
    ax = axes[1]
    ax.barh(y - 0.16, [cls.loc[key, "roc_auc"] for key, _ in PAIR_TARGETS],
            height=0.30, color=COLORS["face"], label="AUROC", zorder=3)
    ax.barh(y + 0.16, [cls.loc[key, "balanced_accuracy"] for key, _ in PAIR_TARGETS],
            height=0.30, color=COLORS["diverse"], label="Balanced accuracy", zorder=3)
    ax.axvline(0.5, color="#9CA3A8", lw=0.8)
    ax.set_xlim(0, 0.82)
    ax.set_yticks(y, [f"{name}   n={int(cls.loc[key, 'n']):,}"
                      for key, name in PAIR_TARGETS])
    ax.invert_yaxis()
    ax.set_xlabel("Up/down classification score")
    ax.legend(loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.0))
    ax.grid(axis="x", color="#E4E7E9", lw=0.7)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.07)
    save(fig, "exp6_delta_tasks.pdf")


def training_history():
    path = (ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
            / "runs/efficientnet_b0/hemoglobin_low/history.csv")
    data = pd.read_csv(path).sort_values("global_epoch")
    fig, ax = plt.subplots(figsize=(7.8, 3.25))
    ax.plot(data.global_epoch, data.train_loss, color=COLORS["face"], lw=1.8,
            label="Training loss")
    ax.plot(data.global_epoch, data.val_loss, color=COLORS["combined"], lw=1.8,
            label="Validation loss")
    head_last = data.loc[data.stage.eq("head"), "global_epoch"].max()
    ax.axvline(head_last + 0.5, color="#777F84", ls="--", lw=1)
    ax.text(head_last + 0.8, ax.get_ylim()[1] * 0.96, "Encoder unfrozen",
            color="#56616E", fontsize=8, va="top")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled Smooth L1 loss")
    ax.set_title("Face-only hemoglobin regression: original holdout run", loc="left")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", color="#E4E7E9", lw=0.7)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.17)
    save(fig, "exp2_hemoglobin_training.pdf")


def predicted_vs_true():
    sources = [
        ("Face-only hemoglobin", ROOT / "study/exp2_face_pretrained_head32_regression"
         / "outputs/20frame/runs/efficientnet_b0/hemoglobin_low/video_predictions.csv",
         "Hemoglobin (g/L)"),
        ("Paired-face hemoglobin change", ROOT / "study/exp6_face_pair_lab_delta"
         / "outputs/runs/hemoglobin_low/pair_predictions.csv", "Hemoglobin change (g/L)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.7))
    for ax, (title, path, label) in zip(axes, sources):
        data = pd.read_csv(path).query("split == 'test'")
        x = data.y_true.to_numpy(float)
        y = data.y_pred.to_numpy(float)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        span = hi - lo
        lo -= 0.04 * span
        hi += 0.04 * span
        ax.scatter(x, y, s=14, color=COLORS["face"], alpha=0.43,
                   edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="#737C80", ls="--", lw=1,
                label="Identity")
        slope, intercept = np.polyfit(x, y, deg=1)
        fit_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(fit_x, slope * fit_x + intercept, color=COLORS["combined"],
                lw=1.8, label="Least-squares fit")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{title}  (n={len(data)})", fontsize=9)
        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.grid(color="#E4E7E9", lw=0.6)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.19, wspace=0.30)
    save(fig, "hemoglobin_predicted_vs_true.pdf")


def tables():
    table_dir = Path(__file__).resolve().parent / "tables"
    table_dir.mkdir(exist_ok=True)
    base = "study/exp2_face_pretrained_head32_regression/outputs/5fold/cv_summary.csv"
    diverse = "study/exp2_face_pretrained_head32_regression/outputs/ablations/patient_diverse_schedule_30_40/5fold/cv_summary.csv"
    video = "study/exp3_video_lab_regression/outputs/5fold/cv_summary.csv"
    classification = "study/exp2_face_pretrained_head32_classification/outputs/5fold/cv_summary.csv"
    a, b, c = (cv(path, "r2") for path in (base, diverse, video))
    auc, bacc = (cv(classification, name) for name in ("roc_auc", "balanced_accuracy"))
    rows = []
    for key, name in TARGETS:
        rows.append(
            f"{name} & {int(a.loc[key, 'videos']):,} & "
            f"{a.loc[key, 'pooled_oof']:.3f} & {b.loc[key, 'pooled_oof']:.3f} & "
            f"{c.loc[key, 'pooled_oof']:.3f} & {auc.loc[key, 'pooled_oof']:.3f} & "
            f"{bacc.loc[key, 'pooled_oof']:.3f} \\\\"
        )
    (table_dir / "exp2_cv_rows.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n", encoding="ascii"
    )

    reg = holdout("study/exp6_face_pair_lab_delta/outputs/metrics_all.csv", "n")
    frame_reg = pd.read_csv(ROOT / "study/exp6_face_pair_lab_delta/outputs/metrics_all.csv")
    frame_cls = pd.read_csv(ROOT / "study/exp6_face_pair_lab_delta_classifacation/outputs/metrics_all.csv")
    frame_reg = frame_reg.loc[frame_reg.split.eq("test")].set_index("target")
    frame_cls = frame_cls.loc[frame_cls.split.eq("test")].set_index("target")
    rows = []
    for key, name in PAIR_TARGETS:
        rows.append(
            f"{name} & {int(reg.loc[key]):,} & {frame_reg.loc[key, 'r2']:.3f} & "
            f"{frame_reg.loc[key, 'pearson_r']:.3f} & {int(frame_cls.loc[key, 'n']):,} & "
            f"{frame_cls.loc[key, 'roc_auc']:.3f} & "
            f"{frame_cls.loc[key, 'balanced_accuracy']:.3f} \\\\"
        )
    (table_dir / "exp6_rows.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n", encoding="ascii"
    )


if __name__ == "__main__":
    style()
    cabg_trajectories()
    regression_cv()
    classification_cv()
    history_comparison()
    pair_delta()
    training_history()
    predicted_vs_true()
    tables()
    print(f"Rendered seven figures and two tables in {OUT.parent}")
