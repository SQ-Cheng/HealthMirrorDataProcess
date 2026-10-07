"""Run prepared native224 Exp4 after the queued architecture controls."""

import fcntl
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PREDECESSOR = HERE.parent / "exp2_face_architecture_ablation/outputs"


def predecessor_complete():
    with (PREDECESSOR / ".queue.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("The preceding architecture queue stopped before completion")
    return True


def main():
    (HERE / "logs").mkdir(exist_ok=True)
    with (HERE / "logs/.monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not predecessor_complete():
            print("[waiting] architecture-control queue active; native224 Exp4 prepared; no GPU allocated", flush=True)
            time.sleep(60)
        subprocess.run([sys.executable, "-u", "-m", "study.exp4.run_all", "--reuse-prepared"], check=True)
        print("[queue-complete] native224 Exp4 training and figures", flush=True)


if __name__ == "__main__":
    main()
