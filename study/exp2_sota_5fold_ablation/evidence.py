"""Snapshot completed parent results without promoting pending runs to evidence."""

import json

import pandas as pd

from study.common.run_video_loss_12h import sha256
from . import config


def write_evidence():
    root = config.OUTPUT/"design_evidence"
    root.mkdir(parents=True, exist_ok=True)
    if (root/"provenance.json").exists():
        if not (root/"parent_metrics.csv").exists():
            raise RuntimeError("Partial design-evidence snapshot")
        # Freeze evidence at design time; a later parent completion is not new rationale.
        return
    hashes, results, pending = {}, [], []
    baseline = config.parent.BASELINE/f"regression/{config.parent.ARCHITECTURE}"
    for name, folder in [("baseline", baseline), *[(name, config.parent.OUTPUT/name) for name in config.parent.VARIANTS]]:
        if name != "baseline" and not (folder/"COMPLETE").exists():
            pending.append(name)
            continue
        for target in config.TARGETS:
            file = folder/f"runs/{target}/metrics.csv"
            hashes[str(file)] = sha256(file)
            frame = pd.read_csv(file)
            for split in ("val", "test"):
                row = frame.loc[frame.split.eq(split)].iloc[0].to_dict()
                row.update(variant=name, target=target)
                results.append(row)
    pd.DataFrame(results).to_csv(root/"parent_metrics.csv", index=False)
    (root/"provenance.json").write_text(json.dumps({"files": hashes, "pending_not_evidence": pending,
        "interpretation": "Exploratory design informed by earlier same-cohort experiments, not an untouched confirmatory test. "
        "New outer-test predictions must never drive automatic protocol selection."}, indent=2)+"\n")
