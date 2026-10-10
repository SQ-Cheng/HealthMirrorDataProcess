"""Real multi-GPU BYOL update and one supervised two-stage handoff, in temporary storage."""

import gc
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from . import config
from .run_all import sha256, run_ssl_subprocess
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler


def run_smoke(records, videos, ssl_index, supervised_index, scalers):
    from study.exp2_face_pretrained_head32_regression import train
    gpus = min(4, torch.cuda.device_count())
    if not gpus: raise RuntimeError("Smoke requires CUDA")
    chosen = videos.drop_duplicates("hospital_id").head(config.SSL_ANCHORS_PER_GPU * gpus).copy()
    with tempfile.TemporaryDirectory(prefix="byol_transfer_smoke_") as temporary:
        root = Path(temporary)
        run_ssl_subprocess(("--smoke-output", str(root / "ssl")))
        encoder = root / "ssl/encoder.pt"
        state = torch.load(encoder, map_location="cpu", weights_only=True)
        assert set(state["train_patients"]) == set(chosen.hospital_id)
        gc.collect(); torch.cuda.empty_cache()
        target = "hemoglobin_low"; table = records[target]
        subset = pd.concat([table.loc[table.split.eq("train") & table.hospital_id.isin(chosen.hospital_id)]
                            .drop_duplicates("clinical_event_id").head(12),
                            table.loc[table.split.ne("train")].groupby("split", group_keys=False).head(2)])
        assert not set(subset.loc[subset.split.ne("train"), "hospital_id"]) & set(chosen.hospital_id)
        with patch.object(train, "TORCH_COMPILE_ENABLED", False), patch.object(train, "TRAIN_NUM_WORKERS", 0), \
             patch.object(train, "EVAL_NUM_WORKERS", 0):
            train.train_task("efficientnet_b0", target, supervised_index, subset, RobustTargetScaler(**scalers[target]),
                             config.reference.WEIGHTS_DIR, str(root / "regression"), head_epochs=1, finetune_epochs=1,
                             max_batches=1, initial_encoder_state_path=str(encoder),
                             train_batch_policy="distinct_lab_views", loss_level="frame")
        saved = torch.load(root / "regression/model.pt", map_location="cpu", weights_only=True)
        assert saved["initial_encoder_sha256"] == sha256(encoder) and saved["loss_level"] == "frame"
    print("[smoke-ok] real BYOL online/EMA teacher update on available GPUs; train-only pool; two-stage regression handoff; saved checkpoints", flush=True)
