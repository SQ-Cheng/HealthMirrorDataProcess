"""Explicitly finish SSL early using the last saved online backbone, not live state."""

import fcntl
import json

import torch

from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model
from . import config
from .byol import atomic_save
from .run_all import preflight, sha256


def promote():
    records, videos, ssl_index, supervised, scalers, manifest = preflight()
    folder = config.OUTPUT / "ssl"; source = folder / "last.pt"; destination = folder / "encoder.pt"
    if destination.exists():
        raise FileExistsError("An exported SSL encoder already exists; will not overwrite its downstream contract")
    if any((config.OUTPUT / "runs").rglob("model.pt")):
        raise RuntimeError("Downstream models already exist")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    contract = sha256(config.OUTPUT / "experiment_manifest.json")
    if checkpoint["contract"] != contract or not 0 < checkpoint["epoch"] <= config.SSL_EPOCHS:
        raise RuntimeError("The SSL checkpoint has a different data/training contract")
    if not checkpoint["history"] or checkpoint["history"][-1]["projection_std"] < 1e-4:
        raise RuntimeError("The saved SSL checkpoint lacks a healthy projection-variance audit")
    prefix = "online_features."
    features = {key[len(prefix):]: value.detach().cpu()
                for key, value in checkpoint["model"].items() if key.startswith(prefix)}
    model, _, _ = build_pretrained_model("efficientnet_b0", config.reference.WEIGHTS_DIR)
    model.features.load_state_dict(features, strict=True)
    if any(not torch.isfinite(value).all() for value in features.values() if value.is_floating_point()):
        raise RuntimeError("Nonfinite online encoder weights")
    source_hash = sha256(source)
    atomic_save({
        "features_state_dict": features, "method": "BYOL", "contract": contract,
        "ssl_epochs": checkpoint["epoch"], "ssl_planned_epochs": config.SSL_EPOCHS,
        "source_step": checkpoint["step"], "source_checkpoint_sha256": source_hash,
        "early_stopped_by_user": True, "train_patients": sorted(set(videos.hospital_id)),
    }, destination)
    transition = {
        "reason": "user requested interruption of BYOL and immediate transition to downstream tasks",
        "source_checkpoint": str(source), "source_checkpoint_sha256": source_hash,
        "completed_ssl_epochs": checkpoint["epoch"], "saved_global_step": checkpoint["step"],
        "planned_ssl_epochs": config.SSL_EPOCHS,
        "unsaved_in_progress_epoch_updates_used": False,
        "encoder": str(destination), "encoder_sha256": sha256(destination),
        "contract": contract, "split_labels_scalers_and_frame_indexes_unchanged": True,
        "ssl_checkpoint_and_history_preserved": True,
    }
    (folder / "transition.json").write_text(json.dumps(transition, indent=2) + "\n")
    print(f"[ssl-promoted] saved_epoch={checkpoint['epoch']} step={checkpoint['step']} "
          f"planned_epochs={config.SSL_EPOCHS}; online features only; data/split unchanged", flush=True)


if __name__ == "__main__":
    with (config.OUTPUT / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        promote()
