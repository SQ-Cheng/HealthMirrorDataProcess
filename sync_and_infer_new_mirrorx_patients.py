#!/usr/bin/env python3
"""Sync new MirrorX patients, then run local inference on pending folders."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from queue import Queue

from tqdm import tqdm
from sync_new_mirrorx_patients import (
    DEFAULT_RETRIES,
    DEFAULT_WORKERS,
    HOST,
    LOCAL_BASE,
    PATIENT_DIR_RE,
    REMOTE_BASE,
    SSHMaster,
    SyncError,
    TransferTask,
    build_transfer_plan,
    check_local_base,
    check_remote_base,
    check_remote_rsync,
    discover_local_mirror_ids,
    discover_remote_mirror_ids,
    format_mirror_ids,
    local_mirror_root,
    parse_mirror_ids,
    remote_mirror_root,
    require_command,
    run_transfers,
)


MIRROR_DIR_RE = re.compile(r"^mirror(\d+)_data$")


def mirror_version_for_id(mirror_id: int) -> str:
    return "1" if mirror_id in {1, 2} else "2"


def patient_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.removeprefix("patient_")
    return (int(suffix) if suffix.isdigit() else -1, path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync newly discovered patient_xxxxxx directories from MirrorX, "
            "then run local inference on the newly synced folders."
        )
    )
    parser.add_argument("--host", default=HOST, help=f"SSH host, default: {HOST}")
    parser.add_argument("--user", default="root", help="SSH user, default: root")
    parser.add_argument("--remote-base", default=REMOTE_BASE, help=f"Remote mirror base, default: {REMOTE_BASE}")
    parser.add_argument(
        "--local-base",
        default=str(LOCAL_BASE),
        help=f"Local mirror base, default: {LOCAL_BASE}",
    )
    parser.add_argument(
        "--mirror-id",
        action="append",
        help="Mirror number to sync/infer, e.g. --mirror-id 1 --mirror-id 6 or --mirror-id 1,6. Default: all remote mirrors.",
    )
    parser.add_argument(
        "--create-missing-local-mirrors",
        action="store_true",
        help="Create missing local mirrorN_data roots when a remote health_mirror_0N exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel rsync worker count, default: {DEFAULT_WORKERS}. Use 1 for serial transfers.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retry attempts for SSH commands and rsync transfers, default: {DEFAULT_RETRIES}.",
    )
    parser.add_argument(
        "--model-choice",
        default="Step",
        choices=["Step"],
        help="Local inference model choice, default: Step.",
    )
    parser.add_argument(
        "--mirror-version",
        default="auto",
        choices=["auto", "1", "2"],
        help="Inference mirror version. Default: auto (mirror1/2 -> 1, others -> 2).",
    )
    parser.add_argument(
        "--infer-legacy-missing",
        action="store_true",
        help=(
            "Also scan selected local mirror roots for previously synced patient folders "
            "that do not have a valid rppg_log.csv, then batch infer them. Default: off."
        ),
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Run inference even if rppg_log.csv already exists and is valid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sync and inference targets without downloading or running inference.",
    )

    args = parser.parse_args()
    try:
        args.mirror_ids = parse_mirror_ids(args.mirror_id)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.retries < 1:
        parser.error("--retries must be >= 1")
    return args


def resolve_sync_targets(args: argparse.Namespace) -> tuple[list[TransferTask], set[int]]:
    local_base = Path(args.local_base).expanduser()
    remote_base = args.remote_base
    ssh = SSHMaster(args.host, args.user, args.retries)

    try:
        require_command("ssh")
        require_command("rsync")
        check_local_base(local_base)

        print("Establishing persistent SSH master connection...")
        ssh.establish()

        print("Checking remote mirror base and rsync availability...")
        check_remote_base(remote_base, ssh)
        check_remote_rsync(ssh)

        print("Discovering mirror directories...")
        local_mirror_ids = discover_local_mirror_ids(local_base)
        remote_mirror_ids = discover_remote_mirror_ids(remote_base, ssh)
        if not local_mirror_ids:
            raise SyncError(f"No local mirrorN_data directories found under {local_base}")
        if not remote_mirror_ids:
            raise SyncError(f"No remote health_mirror_NN directories found under {remote_base}")

        target_mirror_ids = args.mirror_ids if args.mirror_ids is not None else set(remote_mirror_ids)
        missing_remote = target_mirror_ids - remote_mirror_ids
        if missing_remote:
            raise SyncError(
                "Requested mirror roots are missing on the server: "
                + ", ".join(remote_mirror_root(remote_base, mirror_id) for mirror_id in sorted(missing_remote))
            )

        missing_local = target_mirror_ids - local_mirror_ids
        if missing_local and not args.create_missing_local_mirrors:
            raise SyncError(
                "Remote mirror roots exist but the matching local roots are missing: "
                + ", ".join(str(local_mirror_root(local_base, mirror_id)) for mirror_id in sorted(missing_local))
                + "\nCreate them manually, pass --create-missing-local-mirrors, or limit --mirror-id."
            )
        for mirror_id in sorted(missing_local):
            path = local_mirror_root(local_base, mirror_id)
            if args.dry_run:
                print(f"Would create missing local mirror root: {path}")
            else:
                path.mkdir(parents=False, exist_ok=False)
                local_mirror_ids.add(mirror_id)
                print(f"Created missing local mirror root: {path}")

        print(f"Local mirror IDs: {format_mirror_ids(local_mirror_ids)}")
        print(f"Remote mirror IDs: {format_mirror_ids(remote_mirror_ids)}")
        print(f"Mirror IDs selected for sync/inference: {format_mirror_ids(target_mirror_ids)}")

        pending = build_transfer_plan(local_base, remote_base, ssh, target_mirror_ids, missing_local, args.dry_run)
        if not pending:
            print("No new patient directories found.")
            return [], target_mirror_ids

        print(f"Total patient directories queued for sync: {len(pending)}")
        print(f"Worker count: {args.workers}")
        if args.dry_run:
            print("Dry run: no files will be downloaded.")
            return pending, target_mirror_ids

        run_transfers(pending, ssh, args.workers, args.retries)
        print("Sync complete.")
        return pending, target_mirror_ids
    finally:
        ssh.close()


def task_patient_paths(tasks: list[TransferTask]) -> list[Path]:
    return [task.local_root / task.patient_id for task in tasks]


def has_existing_results(path: Path) -> bool:
    log_path = path / "rppg_log.csv"
    if not log_path.is_file():
        return False
    try:
        with log_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return False
            required_columns = {"timestamp", "rppg"}
            if not required_columns.issubset(reader.fieldnames):
                return False

            row_count = 0
            for row in reader:
                try:
                    timestamp = float(row["timestamp"])
                    rppg = float(row["rppg"])
                except (TypeError, ValueError):
                    return False
                if not math.isfinite(timestamp) or not math.isfinite(rppg):
                    return False
                row_count += 1
    except OSError as exc:
        print(f"[Inference] Warning: Cannot read existing result {log_path}: {exc}")
        return False

    return row_count > 0


def discover_uninferred_patient_paths(local_base: Path, mirror_ids: set[int]) -> list[Path]:
    pending: list[Path] = []
    for mirror_id in sorted(mirror_ids):
        mirror_root = local_mirror_root(local_base, mirror_id)
        if not mirror_root.is_dir():
            print(f"[Inference] Warning: local mirror root does not exist, skipping legacy scan: {mirror_root}")
            continue
        for patient_dir in sorted(mirror_root.iterdir(), key=patient_sort_key):
            if not patient_dir.is_dir() or PATIENT_DIR_RE.fullmatch(patient_dir.name) is None:
                continue
            if not has_existing_results(patient_dir):
                pending.append(patient_dir)
    return pending


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def mirror_id_from_patient_path(path: Path) -> int | None:
    match = MIRROR_DIR_RE.fullmatch(path.parent.name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_mirror_version(path: Path, configured: str) -> str:
    if configured != "auto":
        return configured
    mirror_id = mirror_id_from_patient_path(path)
    if mirror_id is None:
        return "1"
    return mirror_version_for_id(mirror_id)


def reset_inference_queues(local_inference: LocalInference) -> None:
    local_inference.preprocess_queue = Queue()
    local_inference.result_queue = Queue()


def run_inference(paths: list[Path], args: argparse.Namespace) -> int:
    if not paths:
        print("[Inference] No patient directories queued.")
        return 0

    import global_vars
    import inference_vars
    from local_inference_new import LocalInference, signal_handler

    signal.signal(signal.SIGINT, signal_handler)
    global_vars.user_interrupt = False

    failures: list[str] = []
    inference_by_version: dict[str, LocalInference] = {}

    print(f"[Inference] Patient directories queued: {len(paths)}")
    for path in tqdm(paths):
        if not path.is_dir():
            failures.append(f"{path}: not a valid directory")
            print(f"[Inference] Error: {path} is not a valid directory.")
            continue
        if not args.force_inference and has_existing_results(path):
            print(f"[Inference] Skipping already inferenced directory: {path}")
            continue

        version = resolve_mirror_version(path, args.mirror_version)
        global_vars.mirror_version = version
        inference_vars.mirror_version = version
        local_inference = inference_by_version.get(version)
        if local_inference is None:
            local_inference = LocalInference(
                model_choice=args.model_choice,
                mirror_version=version,
                data_dir=str(path.parent),
                skip_existing=not args.force_inference,
            )
            inference_by_version[version] = local_inference
        reset_inference_queues(local_inference)

        try:
            print(f"[Inference] Processing directory: {path} (mirror_version={version})")
            local_inference._inference(path=str(path))
        except Exception as exc:
            failures.append(f"{path}: {exc}")
            print(f"[Inference] Error while processing {path}: {exc}", file=sys.stderr)

        if global_vars.user_interrupt:
            print("[Inference] Interrupted by user.")
            return 130

    if failures:
        print("[Inference] Some directories failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("[Inference] Complete.")
    return 0


def main() -> int:
    args = parse_args()
    local_base = Path(args.local_base).expanduser()

    try:
        synced_tasks, target_mirror_ids = resolve_sync_targets(args)
        inference_paths = task_patient_paths(synced_tasks)

        if args.infer_legacy_missing:
            legacy_paths = discover_uninferred_patient_paths(local_base, target_mirror_ids)
            print(f"[Inference] Legacy missing-result directories found: {len(legacy_paths)}")
            inference_paths.extend(legacy_paths)

        inference_paths = dedupe_paths(inference_paths)
        if args.dry_run:
            print(f"Dry run: patient directories queued for inference: {len(inference_paths)}")
            for path in inference_paths:
                print(f"  {path}")
            print("Dry run complete; no inference was run.")
            return 0

        return run_inference(inference_paths, args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except (SyncError, subprocess.SubprocessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
