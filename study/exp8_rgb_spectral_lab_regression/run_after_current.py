"""Wait for the current native-224 five-fold queue, then run both Exp8 grids."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREDECESSOR = HERE.parent / "common/outputs/face224_12h_5fold"


def predecessor_complete():
    with (PREDECESSOR / ".queue.lock").open("r") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not (PREDECESSOR / "COMPLETE").is_file():
            raise RuntimeError("The five-fold predecessor stopped before completion; refusing to start Exp8")
    return True


def main():
    logs = HERE / "logs_autostart"
    logs.mkdir(exist_ok=True)
    with (logs / ".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not predecessor_complete():
            print("[waiting] native224 12h five-fold queue active; no GPU allocated", flush=True)
            time.sleep(60)
        for grid in (4, 16):
            suffix = "_face224" if grid == 4 else "_grid16_face224"
            output = HERE / f"outputs{suffix}"
            if (output / "COMPLETE").is_file():
                manifest = json.loads((output / "experiment_manifest.json").read_text())
                if (manifest["matching_hours"] != 12 or manifest["source_resolution"] != 224
                        or manifest["feature_shape"] != [31, grid, grid]):
                    raise RuntimeError(f"Completed Exp8 result has the wrong protocol: {output}")
                print(f"[reuse-complete] Exp8 12h grid={grid}", flush=True)
                continue
            env = os.environ.copy()
            env.update(HEALTHMIRROR_FACE_SOURCE="face224", EXP8_VARIANT="ntire2022",
                       EXP8_GRID_SIZE=str(grid))
            for key in ("EXP8_BASE_OUTPUT", "EXP8_INDEX_PATH", "EXP8_CACHE", "EXP8_OUTPUT"):
                env.pop(key, None)
            run_logs = HERE / f"logs{suffix}"
            run_logs.mkdir(exist_ok=True)
            print(f"[experiment-start] Exp8 12h grid={grid}", flush=True)
            with (run_logs / "run.log").open("a") as log:
                child = subprocess.Popen(
                    [sys.executable, "-u", "-m", "study.exp8_rgb_spectral_lab_regression.train"],
                    cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if child.wait():
                    raise RuntimeError(f"Exp8 12h grid={grid} failed")
        (logs / "COMPLETE").write_text("12h grid4/grid16 training and figures completed\n")
        print("[queue-complete] Exp8 native224 12h, both feature grids", flush=True)


if __name__ == "__main__":
    main()
