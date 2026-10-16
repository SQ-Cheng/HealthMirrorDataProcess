"""Explicit one-off provenance migration before any five-fold IR job was started."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json

from study.common.run_video_loss_12h import sha256
from study.exp2_sota_5fold_ablation import config
from study.exp2_sota_5fold_ablation.run import code_hashes, complete

ADDED = {"compile_dynamic": False, "compile_fullgraph": True,
         "farl_attention": "local_mask_no_attribute_assignment"}
ALLOWED_CODE = {"study/exp2_sota_ablation/config.py", "study/exp2_sota_ablation/online.py",
    "study/exp2_sota_5fold_ablation/config.py", "study/exp2_sota_5fold_ablation/run.py",
    "study/common/ir_compile.py"}


def validate_policy(old, current):
    expected = deepcopy(old)
    expected["protocols"]["c08_ir_tail"].update(ADDED)
    expected["code"] = current["code"]
    if expected != current:
        raise RuntimeError("Migration may change only the not-started IR compiler policy/code")
    changed = {key for key in set(old["code"]) | set(current["code"])
               if old["code"].get(key) != current["code"].get(key)}
    if not changed.issubset(ALLOWED_CODE):
        raise RuntimeError(f"Unexpected training-source changes: {changed-ALLOWED_CODE}")


def migrate():
    protocol = config.OUTPUT/"protocol.json"
    old = json.loads(protocol.read_text())
    current = {**deepcopy(old), "protocols": {name: config.settings(name) for name in config.VARIANTS}, "code": code_hashes()}
    if current == old:
        print("[migration] already current")
        return
    validate_policy(old, current)
    if (config.OUTPUT/"c08_ir_tail").exists():
        raise RuntimeError("C08 output exists; do not silently mix IR implementations")
    if sha256(config.SPLITS/"manifest.json") != old["split_contract"]:
        raise RuntimeError("Patient split changed")
    manifest = json.loads((config.SPLITS/"manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(config.SPLITS/name) != digest:
            raise RuntimeError("A saved split/distribution file changed")
    old_contract = sha256(protocol)
    text = json.dumps(current, indent=2)+"\n"
    new_contract = hashlib.sha256(text.encode()).hexdigest()
    preserved = []
    for name in config.VARIANTS:
        for marker in (config.OUTPUT/name).glob("*/fold*/complete.json"):
            if config.settings(name)["online_epochs"] or old["protocols"][name] != current["protocols"][name]:
                raise RuntimeError("Cannot preserve a completed affected training job")
            if not complete(marker.parent, old_contract):
                raise RuntimeError(f"Invalid completed artifact: {marker}")
            preserved.append(marker)
    archive = config.OUTPUT/"protocol_migrations"/old_contract
    if archive.exists():
        raise RuntimeError("Partial migration archive exists; inspect before retrying")
    archive.mkdir(parents=True)
    (archive/"protocol_before.json").write_text(protocol.read_text())
    for marker in preserved:
        relative = str(marker.relative_to(config.OUTPUT))
        backup = archive/relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(marker.read_text())
        state = json.loads(marker.read_text())
        state.update(contract=new_contract, original_contract=old_contract,
            compatibility_reason="completed frozen-only job unchanged; IR compiler fix is unused by this job")
        marker.write_text(json.dumps(state)+"\n")
    protocol.write_text(text)
    report = {"old_contract": old_contract, "new_contract": new_contract, "changed_code": sorted(ALLOWED_CODE),
        "preserved_complete_jobs": [str(path.relative_to(config.OUTPUT)) for path in preserved],
        "model_and_predictions_modified": False, "split_modified": False, "c08_started_before_fix": False,
        "checkpoint_origin_contract": "retained in model.pt and complete.json.original_contract"}
    (archive/"migration.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"[migration] {len(preserved)} completed fold jobs preserved, split/models/predictions unchanged")


def migrate_old_a07():
    from study.exp2_sota_ablation import config as original
    root = original.OUTPUT/"a07_ir_tail_adaptation"
    protocol = root/"protocol.json"
    old = json.loads(protocol.read_text())
    new = {**old, "settings": original.get_variant("a07_ir_tail_adaptation")}
    if new == old:
        return
    expected = deepcopy(old)
    expected["settings"].update(ADDED)
    if expected != new or list(root.glob("runs/*/stage_ir_tail_best.pt")) or list(root.glob("runs/*/job_complete.json")):
        raise RuntimeError("Old A07 already has IR results or a different policy change")
    backup = root/"compiler_fix"
    backup.mkdir(exist_ok=False)
    (backup/"protocol_before.json").write_text(protocol.read_text())
    protocol.write_text(json.dumps(new, indent=2)+"\n")
    (backup/"status.json").write_text(json.dumps({"status": "stopped_not_restarted", "ir_epochs_completed": 0,
        "policy": ADDED, "frozen_checkpoints_preserved": True})+"\n")
    print("[migration] old A07 compiler policy updated; experiment remains stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply:
        parser.error("Explicit --apply required; both coordinators must be stopped")
    with (config.OUTPUT/".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (config.parent.OUTPUT/".queue.lock").open("w") as original_lock:
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            migrate()
            migrate_old_a07()


if __name__ == "__main__":
    main()
