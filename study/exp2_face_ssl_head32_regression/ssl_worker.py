"""Separate-process BYOL stage, freeing all CUDA state before downstream jobs."""

from . import config
from .run_all import preflight, sha256
from .byol import pretrain
import torch
import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-output")
    args = parser.parse_args()
    records, videos, index, supervised, scalers, manifest = preflight()
    contract = sha256(config.OUTPUT / "experiment_manifest.json")
    encoder = config.OUTPUT / "ssl/encoder.pt"
    if args.smoke_output:
        chosen = videos.drop_duplicates("hospital_id").head(256).copy()
        pretrain(index, chosen, Path(args.smoke_output), "temporary_smoke", epochs=1, max_steps=1)
    elif encoder.exists():
        saved = torch.load(encoder, map_location="cpu", weights_only=True)
        if saved["contract"] != contract:
            raise RuntimeError("Saved BYOL initialization differs from this experiment")
        if saved["ssl_epochs"] != config.SSL_EPOCHS:
            transition_path = encoder.with_name("transition.json")
            if not saved.get("early_stopped_by_user") or not transition_path.exists():
                raise RuntimeError("Incomplete BYOL encoder was not explicitly accepted")
            transition = json.loads(transition_path.read_text())
            if transition["contract"] != contract or transition["encoder_sha256"] != sha256(encoder):
                raise RuntimeError("Early-stop export lineage is invalid")
        print(f"[reuse-ssl] accepted train-only BYOL checkpoint; actual epochs={saved['ssl_epochs']}", flush=True)
    else:
        pretrain(index, videos, config.OUTPUT / "ssl", contract)
