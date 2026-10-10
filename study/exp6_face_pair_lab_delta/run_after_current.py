"""Wait for the active Exp2 ten-target queue, then overwrite the Exp6 main."""

import fcntl
import json
import subprocess
import sys
import time

from .run_full_data import OUTPUT, LOGS, sha256
from . import config


PREDECESSOR = config.EXP_DIR.parent / "exp2_face_pretrained_head32_regression/outputs/ablations/preoperative_nearest_unlimited_face224"


def predecessor_complete():
    path = PREDECESSOR / ".queue.lock"
    if not path.exists(): raise RuntimeError("Expected Exp2 predecessor lock is missing")
    with path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("Exp2 stopped without successful completion; Exp6 was not started")
    return True


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    status = LOGS / "autostart_status.json"
    with (LOGS / ".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = sha256(PREDECESSOR / "experiment_manifest.json")
        state = {"status": "waiting", "predecessor": str(PREDECESSOR), "predecessor_contract_sha256": contract,
                 "output": str(OUTPUT), "overwrite": True, "tasks": 11, "hours": 24, "lab_pairs_per_batch": 12}
        status.write_text(json.dumps(state, indent=2) + "\n")
        try:
            while not predecessor_complete():
                print("[waiting] Exp2 supplementary/preoperative queue active; Exp6 allocates no GPU", flush=True)
                time.sleep(60)
            if sha256(PREDECESSOR / "experiment_manifest.json") != contract:
                raise RuntimeError("Exp2 predecessor contract changed while waiting")
            state["status"] = "training"; status.write_text(json.dumps(state, indent=2) + "\n")
            print(f"[launch] latest full lab table; overwrite {OUTPUT}; eleven models on four GPUs; log={LOGS/'run.log'}", flush=True)
            with (LOGS / "run.log").open("w", buffering=1) as log:
                subprocess.run([sys.executable, "-u", "-m", "study.exp6_face_pair_lab_delta.run_full_data",
                                "--overwrite", "--lab-pairs-per-batch", "12"],
                               cwd=config.REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            state["status"] = "complete"
            print("[queue-complete] Exp6 eleven full-data native224 models and figures", flush=True)
        except Exception as error:
            state.update(status="failed", error=str(error))
            raise
        finally:
            status.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
