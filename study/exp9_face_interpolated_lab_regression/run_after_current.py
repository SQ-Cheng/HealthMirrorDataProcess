"""Wait for both current Exp2 mains, then train prepared Exp9 on four GPUs."""

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from . import config
from .build_dataset import sha256
from .run_all import preflight


PREDECESSOR=config.STUDY/"common/outputs/face_main_24h_frame_loss"


def predecessor_complete():
    with (PREDECESSOR/".queue.lock").open("r") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if not (PREDECESSOR/"COMPLETE").is_file():
            raise RuntimeError("Current Exp2 queue stopped without successful completion")
    roots=[config.SOURCE,config.STUDY/"exp2_face_pretrained_head32_classification/outputs/face224"]
    if not all((root/"COMPLETE").exists() for root in roots):raise RuntimeError("An Exp2 main is incomplete")
    return True


def main():
    config.LOGS.mkdir(exist_ok=True)
    with (config.LOGS/".monitor.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight()
        path=config.OUTPUT/"experiment_manifest.json";contract=sha256(path)
        (config.OUTPUT/"autostart_status.json").write_text(json.dumps({"status":"waiting","predecessor":str(PREDECESSOR),"prepared_contract_sha256":contract},indent=2)+"\n")
        while not predecessor_complete():
            print("[waiting] Exp2 classification/regression main queue active; Exp9 preoperative-nearest/postoperative-interpolation ready; no GPU allocated",flush=True)
            time.sleep(60)
        if sha256(path)!=contract:raise RuntimeError("Exp9 data contract changed while waiting")
        preflight()
        (config.OUTPUT/"autostart_status.json").write_text(json.dumps({"status":"training","predecessor":str(PREDECESSOR),"prepared_contract_sha256":contract},indent=2)+"\n")
        subprocess.run([sys.executable,"-u","-m","study.exp9_face_interpolated_lab_regression.run_all","--train"],check=True,cwd=config.STUDY.parent)
        (config.OUTPUT/"autostart_status.json").write_text(json.dumps({"status":"complete","prepared_contract_sha256":contract},indent=2)+"\n")
        print("[queue-complete] Exp9 models and figures",flush=True)


if __name__=="__main__":main()
