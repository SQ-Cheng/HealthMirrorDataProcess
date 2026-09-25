"""Draw the actual EfficientNet-B0 parameter-freezing map for Exp2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .config import WEIGHTS_DIR
from .models import (
    build_pretrained_model,
    freeze_encoder,
    parameter_counts,
    unfreeze_all,
    unfreeze_efficientnet_tail,
)


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "outputs/ablations/figures/efficientnet_b0_unfreeze_map.png"
)
TRAIN_COLOR = "#207A60"
FROZEN_COLOR = "#D8DFDE"
TEXT_COLOR = "#24312F"


def _group_trainable(groups):
    return [all(parameter.requires_grad for parameter in group.parameters())
            for group in groups]


def plot_unfreeze_map(output_path=OUTPUT_PATH):
    model, head, _ = build_pretrained_model("efficientnet_b0", WEIGHTS_DIR)
    groups = [*model.features, head]
    module_labels = [
        "features[0]  Stem",
        *[f"features[{index}]  MBConv group" for index in range(1, 8)],
        "features[8]  Final projection",
        "classifier  32-dim head",
    ]
    parameter_counts_by_group = [sum(p.numel() for p in group.parameters())
                                 for group in groups]
    backbone_count = sum(parameter_counts_by_group[:-1])

    freeze_encoder(model, head)
    head_only = _group_trainable(groups)
    unfreeze_all(model)
    full_finetune = _group_trainable(groups)
    frozen_modules = unfreeze_efficientnet_tail(model, head)
    tail_finetune = _group_trainable(groups)
    tail_count = sum(count for count, trainable in zip(
        parameter_counts_by_group[:-1], tail_finetune[:-1]
    ) if trainable)
    assert len(frozen_modules) == 7
    assert head_only == [False] * 9 + [True]
    assert all(full_finetune)
    assert tail_finetune == [False] * 7 + [True] * 3
    assert sum(parameter_counts_by_group) == parameter_counts(model)[0]

    figure, axis = plt.subplots(figsize=(16, 8.5))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 16)
    axis.set_ylim(-1.8, 11.5)
    axis.axis("off")

    axis.text(0.3, 10.8, "Module (forward order)", color=TEXT_COLOR,
              fontsize=11, weight="bold")
    axis.text(5.7, 10.8, "Parameters", color=TEXT_COLOR,
              fontsize=11, weight="bold", ha="right")
    columns = (
        (6.2, "Stage 1\nHead training", head_only),
        (9.3, "Stage 2\nFull fine-tune", full_finetune),
        (12.4, "Stage 2\nTail-only fine-tune", tail_finetune),
    )
    for x, heading, _ in columns:
        axis.text(x + 1.25, 11.0, heading, color=TEXT_COLOR,
                  fontsize=11, weight="bold", ha="center", va="center")

    for index, (label, count) in enumerate(zip(module_labels, parameter_counts_by_group)):
        y = 9.5 - index
        axis.add_patch(Rectangle((0.15, y - 0.42), 15.55, 0.84,
                                 facecolor="#F5F7F6" if index % 2 == 0 else "white",
                                 edgecolor="none"))
        axis.text(0.3, y, label, va="center", color=TEXT_COLOR, fontsize=11)
        axis.text(5.7, y, f"{count:,}", va="center", ha="right",
                  color="#52605D", fontsize=10)
        for x, _, status in columns:
            trainable = status[index]
            axis.add_patch(Rectangle((x, y - 0.31), 2.5, 0.62,
                                     facecolor=TRAIN_COLOR if trainable else FROZEN_COLOR,
                                     edgecolor="none"))
            axis.text(x + 1.25, y, "TRAINABLE" if trainable else "FROZEN",
                      ha="center", va="center", fontsize=9, weight="bold",
                      color="white" if trainable else "#53615E")

    head_count = parameter_counts_by_group[-1]
    axis.text(0.3, -0.75,
              f"Full stage 2: {backbone_count + head_count:,} trainable parameters"
              f"  |  Tail-only stage 2: {tail_count + head_count:,} trainable parameters",
              color=TEXT_COLOR, fontsize=11, weight="bold")
    axis.text(0.3, -1.32,
              f"Tail-only unfreezes {tail_count:,} / {backbone_count:,} backbone parameters"
              f" ({100 * tail_count / backbone_count:.1f}%). Frozen blocks keep BatchNorm"
              " statistics fixed.", color="#52605D", fontsize=10)
    figure.suptitle("EfficientNet-B0: which modules are fine-tuned?",
                     fontsize=18, weight="bold", y=0.98, color=TEXT_COLOR)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_unfreeze_map())
