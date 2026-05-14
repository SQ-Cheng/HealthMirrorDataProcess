#!/usr/bin/env python3
"""High-throughput rsync downloader for HealthMirror patient directories."""

from __future__ import annotations

import argparse
import concurrent.futures
import getpass
import hashlib
import os
import posixpath
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import pexpect
except ImportError:
    pexpect = None


HOST = "47.93.46.224"
USER = "root"
REMOTE_BASE = "/root/health_mirror"
REMOTE_MIRROR_TEMPLATE = "health_mirror_{mirror_id:02d}"
LOCAL_BASE = Path("/root/shared/HealthMirrorDataset")
LOCAL_MIRROR_TEMPLATE = "mirror{mirror_id}_data"
PATIENT_DIR_RE = re.compile(r"^patient_\d{6}$")
REMOTE_MIRROR_RE = re.compile(r"^health_mirror_(\d{2})$")
LOCAL_MIRROR_RE = re.compile(r"^mirror(\d+)_data$")
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
CONNECT_TIMEOUT_SECONDS = 20
COMMAND_TIMEOUT_SECONDS = 120
CONTROL_PERSIST = "30m"
RSYNC_TIMEOUT_SECONDS = 3600
TRANSIENT_RSYNC_CODES = {10, 11, 12, 14, 20, 23, 24, 30, 35, 255}


class SyncError(RuntimeError):
    """Raised when the sync cannot continue safely."""


@dataclass(frozen=True)
class TransferTask:
    mirror_id: int
    patient_id: str
    local_root: Path
    remote_root: str

    @property
    def label(self) -> str:
        return f"mirror{self.mirror_id}/{self.patient_id}"


@dataclass(frozen=True)
class TransferResult:
    task: TransferTask
    attempts: int


class SSHMaster:
    """Persistent OpenSSH ControlMaster connection shared by ssh and rsync."""

    def __init__(
        self,
        host: str,
        user: str,
        retries: int,
        control_persist: str = CONTROL_PERSIST,
    ) -> None:
        self.host = host
        self.user = user
        self.retries = retries
        self.control_persist = control_persist
        token = hashlib.sha1(f"{user}@{host}".encode("utf-8")).hexdigest()[:12]
        self.control_dir = Path(tempfile.mkdtemp(prefix="hm_sync_ssh_", dir="/tmp"))
        self.control_path = self.control_dir / f"cm_{token}.sock"
        self._password: str | None = None
        self._lock = threading.Lock()

    @property
    def remote(self) -> str:
        return f"{self.user}@{self.host}"

    def _options(self, batch_mode: bool) -> list[str]:
        options = [
            "-o",
            f"ConnectTimeout={CONNECT_TIMEOUT_SECONDS}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "NumberOfPasswordPrompts=1",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPersist={self.control_persist}",
            "-o",
            f"ControlPath={self.control_path}",
        ]
        if batch_mode:
            options.extend(["-o", "BatchMode=yes"])
        return options

    def ssh_command(self, batch_mode: bool = True) -> list[str]:
        return ["ssh", *self._options(batch_mode), self.remote]

    def rsync_ssh_command(self) -> str:
        return shlex.join(["ssh", *self._options(batch_mode=True)])

    def establish(self) -> None:
        with self._lock:
            if self.is_alive():
                return
            if self._establish_with_key_auth():
                print("SSH master connection established with key/agent authentication.")
                return
            self._establish_with_password()

    def reestablish(self) -> None:
        with self._lock:
            self.close(remove_dir=False)
            if self._establish_with_key_auth():
                print("SSH master connection re-established with key/agent authentication.")
                return
            self._establish_with_password()

    def is_alive(self) -> bool:
        if not self.control_path.exists():
            return False
        check = subprocess.run(
            ["ssh", "-S", str(self.control_path), "-O", "check", self.remote],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        return check.returncode == 0

    def _master_args(self, batch_mode: bool) -> list[str]:
        return ["ssh", "-fN", "-M", *self._options(batch_mode), self.remote]

    def _establish_with_key_auth(self) -> bool:
        proc = subprocess.run(
            self._master_args(batch_mode=True),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        return proc.returncode == 0

    def _establish_with_password(self) -> None:
        if pexpect is None:
            raise SyncError(
                "SSH key authentication failed and pexpect is not installed. "
                "Install pexpect for password fallback or configure SSH key authentication."
            )
        if self._password is None:
            self._password = getpass.getpass(f"Password for {self.remote}: ")

        child = pexpect.spawn(
            "ssh",
            self._master_args(batch_mode=False)[1:],
            encoding="utf-8",
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        output_parts: list[str] = []
        while True:
            index = child.expect(
                [
                    r"(?i)are you sure you want to continue connecting",
                    r"(?i)(?:password|passphrase).*:",
                    r"(?i)permission denied",
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                ]
            )
            output_parts.append(child.before)
            if index == 0:
                child.sendline("yes")
            elif index == 1:
                child.sendline(self._password)
            elif index == 2:
                output_parts.append(child.after)
                child.close(force=True)
                raise SyncError("SSH authentication failed. Please check the password and try again.")
            elif index == 3:
                child.close()
                if child.exitstatus == 0 or self.is_alive():
                    print("SSH master connection established with password authentication.")
                    return
                raise SyncError("Failed to establish SSH master connection:\n" + "".join(output_parts).strip())
            else:
                child.close(force=True)
                raise SyncError("Timed out while establishing SSH master connection.")

    def run(self, remote_command: str, retries: int | None = None) -> str:
        attempts = retries if retries is not None else self.retries
        last_output = ""
        for attempt in range(1, attempts + 1):
            self.establish()
            try:
                proc = subprocess.run(
                    [*self.ssh_command(batch_mode=True), remote_command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=COMMAND_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                proc = subprocess.CompletedProcess(exc.cmd, 124, exc.stdout or "", exc.stderr or "timeout")

            output = (proc.stdout or "") + (proc.stderr or "")
            last_output = output.strip()
            if proc.returncode == 0:
                return proc.stdout
            if "permission denied" in output.lower():
                raise SyncError("SSH authentication failed. Please check credentials or SSH key configuration.")
            if attempt < attempts:
                print(f"SSH command failed on attempt {attempt}/{attempts}; reconnecting and retrying...")
                self.reestablish()
                time.sleep(attempt * 3)

        raise SyncError("SSH command failed after several retries. Last output:\n" + (last_output or "<no output>"))

    def close(self, remove_dir: bool = True) -> None:
        subprocess.run(
            ["ssh", "-S", str(self.control_path), "-O", "exit", self.remote],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if remove_dir:
            shutil.rmtree(self.control_dir, ignore_errors=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SyncError(f"Required command is missing from PATH: {name}")


def local_mirror_root(local_base: Path, mirror_id: int) -> Path:
    return local_base / LOCAL_MIRROR_TEMPLATE.format(mirror_id=mirror_id)


def remote_mirror_root(remote_base: str, mirror_id: int) -> str:
    return posixpath.join(remote_base, REMOTE_MIRROR_TEMPLATE.format(mirror_id=mirror_id))


def check_local_base(local_base: Path) -> None:
    if not local_base.is_dir():
        raise SyncError(f"Local data base directory does not exist or is not a directory: {local_base}")


def check_remote_base(remote_base: str, ssh: SSHMaster) -> None:
    ssh.run(f"test -d {shlex.quote(remote_base)}")


def check_remote_rsync(ssh: SSHMaster) -> None:
    ssh.run("command -v rsync >/dev/null 2>&1")


def discover_local_mirror_ids(local_base: Path) -> set[int]:
    mirror_ids: set[int] = set()
    for path in local_base.iterdir():
        if not path.is_dir():
            continue
        match = LOCAL_MIRROR_RE.fullmatch(path.name)
        if match:
            mirror_ids.add(int(match.group(1)))
    return mirror_ids


def discover_remote_mirror_ids(remote_base: str, ssh: SSHMaster) -> set[int]:
    command = (
        "find "
        + shlex.quote(remote_base)
        + " -maxdepth 1 -mindepth 1 -type d -name 'health_mirror_[0-9][0-9]' -printf '%f\\n'"
    )
    output = ssh.run(command)
    mirror_ids: set[int] = set()
    for line in output.splitlines():
        match = REMOTE_MIRROR_RE.fullmatch(line.strip())
        if match:
            mirror_ids.add(int(match.group(1)))
    return mirror_ids


def list_local_patients(local_root: Path) -> set[str]:
    return {
        path.name
        for path in local_root.iterdir()
        if path.is_dir() and PATIENT_DIR_RE.fullmatch(path.name)
    }


def list_remote_patients(remote_root: str, ssh: SSHMaster) -> set[str]:
    command = (
        "find "
        + shlex.quote(remote_root)
        + " -maxdepth 1 -mindepth 1 -type d -name 'patient_[0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\\n'"
    )
    output = ssh.run(command)
    return {line.strip() for line in output.splitlines() if PATIENT_DIR_RE.fullmatch(line.strip())}


def build_rsync_args(task: TransferTask, ssh: SSHMaster) -> list[str]:
    remote_path = posixpath.join(task.remote_root, task.patient_id)
    remote_spec = f"{ssh.remote}:{shlex.quote(remote_path)}"
    return [
        "rsync",
        "-az",
        "--partial",
        "--append-verify",
        "--info=progress2",
        "--human-readable",
        "--outbuf=L",
        "--timeout",
        str(RSYNC_TIMEOUT_SECONDS),
        "-e",
        ssh.rsync_ssh_command(),
        remote_spec,
        str(task.local_root),
    ]


def stream_process_output(proc: subprocess.Popen[str], label: str, print_lock: threading.Lock) -> str:
    assert proc.stdout is not None
    chunks: list[str] = []
    current: list[str] = []
    while True:
        char = proc.stdout.read(1)
        if char == "" and proc.poll() is not None:
            break
        if not char:
            continue
        chunks.append(char)
        if char in {"\n", "\r"}:
            line = "".join(current).strip()
            current.clear()
            if line and ("%" in line or line.startswith("sent ") or line.startswith("total size")):
                with print_lock:
                    print(f"[{label}] {line}", flush=True)
        else:
            current.append(char)

    tail = "".join(current).strip()
    if tail:
        with print_lock:
            print(f"[{label}] {tail}", flush=True)
    return "".join(chunks)


def transfer_patient(
    task: TransferTask,
    ssh: SSHMaster,
    retries: int,
    print_lock: threading.Lock,
) -> TransferResult:
    last_output = ""
    for attempt in range(1, retries + 1):
        with print_lock:
            print(f"[{task.label}] rsync attempt {attempt}/{retries}", flush=True)
        proc = subprocess.Popen(
            build_rsync_args(task, ssh),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "RSYNC_RSH": ssh.rsync_ssh_command()},
        )
        output = stream_process_output(proc, task.label, print_lock)
        code = proc.wait()
        last_output = output.strip()
        if code == 0:
            return TransferResult(task=task, attempts=attempt)
        if "permission denied" in output.lower():
            raise SyncError(f"{task.label}: authentication failed during rsync.")
        if code in TRANSIENT_RSYNC_CODES and attempt < retries:
            with print_lock:
                print(f"[{task.label}] rsync failed with code {code}; reconnecting and retrying...", flush=True)
            ssh.reestablish()
            time.sleep(attempt * 5)
            continue
        break

    raise SyncError(f"{task.label}: rsync failed after {retries} attempts. Last output:\n{last_output or '<no output>'}")


def parse_mirror_ids(values: list[str] | None) -> set[int] | None:
    if not values:
        return None

    mirror_ids: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise argparse.ArgumentTypeError(f"Mirror id must be numeric, got: {part}")
            mirror_id = int(part)
            if mirror_id <= 0:
                raise argparse.ArgumentTypeError(f"Mirror id must be positive, got: {part}")
            mirror_ids.add(mirror_id)
    return mirror_ids


def format_mirror_ids(mirror_ids: set[int]) -> str:
    return ", ".join(str(mirror_id) for mirror_id in sorted(mirror_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-throughput rsync sync for patient_xxxxxx directories missing locally."
    )
    parser.add_argument("--host", default=HOST, help=f"SSH host, default: {HOST}")
    parser.add_argument("--user", default=USER, help=f"SSH user, default: {USER}")
    parser.add_argument("--remote-base", default=REMOTE_BASE, help=f"Remote mirror base, default: {REMOTE_BASE}")
    parser.add_argument(
        "--local-base",
        default=str(LOCAL_BASE),
        help=f"Local mirror base, default: {LOCAL_BASE}",
    )
    parser.add_argument(
        "--mirror-id",
        action="append",
        help="Mirror number to sync, e.g. --mirror-id 1 --mirror-id 6 or --mirror-id 1,6. Default: all remote mirrors.",
    )
    parser.add_argument(
        "--create-missing-local-mirrors",
        action="store_true",
        help="Create missing local mirrorN_data roots when a remote health_mirror_0N exists.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list new patient directories.")
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


def build_transfer_plan(
    local_base: Path,
    remote_base: str,
    ssh: SSHMaster,
    target_mirror_ids: set[int],
    missing_local: set[int],
    dry_run: bool,
) -> list[TransferTask]:
    pending: list[TransferTask] = []
    for mirror_id in sorted(target_mirror_ids):
        local_root = local_mirror_root(local_base, mirror_id)
        remote_root = remote_mirror_root(remote_base, mirror_id)
        print(f"Comparing mirror {mirror_id}: {remote_root} -> {local_root}")
        local_patients = set() if mirror_id in missing_local and dry_run else list_local_patients(local_root)
        remote_patients = list_remote_patients(remote_root, ssh)
        new_patients = sorted(remote_patients - local_patients)

        print(f"  Local patient directories: {len(local_patients)}")
        print(f"  Remote patient directories: {len(remote_patients)}")
        print(f"  New patient directories to sync: {len(new_patients)}")
        for patient_id in new_patients:
            print(f"  {patient_id}")
            pending.append(
                TransferTask(
                    mirror_id=mirror_id,
                    patient_id=patient_id,
                    local_root=local_root,
                    remote_root=remote_root,
                )
            )
    return pending


def run_transfers(tasks: list[TransferTask], ssh: SSHMaster, workers: int, retries: int) -> None:
    print_lock = threading.Lock()
    total = len(tasks)
    completed = 0
    failed: list[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(transfer_patient, task, ssh, retries, print_lock): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                completed += 1
                with print_lock:
                    print(
                        f"[overall] {completed}/{total} complete: {result.task.label} "
                        f"({result.attempts} attempt(s))",
                        flush=True,
                    )
            except Exception as exc:
                completed += 1
                failed.append(f"{task.label}: {exc}")
                with print_lock:
                    print(f"[overall] {completed}/{total} failed: {task.label}", flush=True)

    if failed:
        raise SyncError("Some transfers failed:\n" + "\n".join(failed))


def main() -> int:
    args = parse_args()
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
        print(f"Mirror IDs selected for sync: {format_mirror_ids(target_mirror_ids)}")

        pending = build_transfer_plan(local_base, remote_base, ssh, target_mirror_ids, missing_local, args.dry_run)
        if not pending:
            print("No new patient directories found.")
            return 0

        print(f"Total patient directories queued: {len(pending)}")
        print(f"Worker count: {args.workers}")
        if args.dry_run:
            print("Dry run complete; no files were downloaded.")
            return 0

        run_transfers(pending, ssh, args.workers, args.retries)
        print("Sync complete.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except (SyncError, subprocess.SubprocessError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
