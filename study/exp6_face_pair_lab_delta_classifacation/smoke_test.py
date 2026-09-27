"""Check cached paired JPEGs, five-view preprocessing, and binary gradients."""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from study.exp2_face_history_head32_regression.frame_index import FrameOffsetIndex
from study.exp6_face_pair_lab_delta.config import CACHE_DIR, OUTPUT_DIR, VIEWS
from study.exp6_face_pair_lab_delta.data import PairedFrameDataset
from study.exp6_face_pair_lab_delta.models import build_model, freeze_backbone
from study.exp6_face_pair_lab_delta.train import _prepare


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA")
    records = pd.read_csv(
        OUTPUT_DIR / "task_records" / "hemoglobin_low.csv",
        dtype={"hospital_id": str},
    )
    records = records.loc[records.split.eq("train") &
                          ~np.isclose(records.raw_delta, 0.0, atol=1e-12, rtol=0)]
    down = records.loc[records.raw_delta.lt(0)].head(1)
    up = records.loc[records.raw_delta.gt(0)].head(1)
    sample = pd.concat([down, up], ignore_index=True)
    sample["scaled_delta"] = sample.raw_delta.gt(0).astype(np.float32)
    index = FrameOffsetIndex.load(CACHE_DIR / "frame_offsets.npz")
    dataset = PairedFrameDataset(index, sample, views=VIEWS, expand_views=True)
    items = [dataset[0], dataset[20]]
    device = torch.device("cuda:0")
    first = _prepare(torch.stack([item[0] for item in items]),
                     torch.stack([item[4] for item in items]), device)
    second = _prepare(torch.stack([item[1] for item in items]),
                      torch.stack([item[4] for item in items]), device)
    labels = torch.stack([item[2] for item in items]).repeat_interleave(5).to(device)
    weights = torch.stack([item[5] for item in items]).repeat_interleave(5).to(device)
    model, _ = build_model("shared")
    model = model.to(device)
    freeze_backbone(model)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(first, second).squeeze(1)
        values = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(1.0, device=device), reduction="none",
        )
        loss = (values * weights).sum() / weights.sum()
    loss.backward()
    if (not torch.isfinite(loss) or logits.shape != (10,)
            or not any(parameter.grad is not None for parameter in model.head.parameters())):
        raise AssertionError("Classification forward/backward check failed")
    dataset.close()
    print(f"[smoke-ok] decoded_pairs=2 augmented_pairs=10 "
          f"labels={labels.unique().tolist()} loss={loss.item():.5f}")


if __name__ == "__main__":
    main()
