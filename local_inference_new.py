import global_vars
import inference_vars
from queue import Queue
from model.step import Step
import preprocess.video2frame as v2f

import threading
import time
import pandas as pd
import numpy as np
import os
import signal
from tqdm import tqdm

mirror_version = global_vars.mirror_version

def signal_handler(sig, frame):
    global_vars.user_interrupt = True

class LocalInference:
    def __init__(self, model_choice="Step", mirror_version="1", data_dir=None, skip_existing=True):
        self.model_choice = model_choice
        self.mirror_version = mirror_version
        self.data_dir = data_dir
        self.skip_existing = skip_existing
        self.model = None
        self.preprocess_queue = Queue()
        self.result_queue = Queue()
        self.video2frame = None

    def _init_model(self):
        if self.model_choice == "Step":
            model = Step(
                model_path="./model/models/onnx/step.onnx",
                state_path="./model/models/onnx/state.pkl",
                dt=1 / 30
            )
        return model

    def _ensure_model(self):
        if self.model is None:
            self.model = self._init_model()

    def _has_existing_results(self, path=None):
        log_path = os.path.join(path, "rppg_log.csv")
        if not os.path.isfile(log_path):
            return False
        try:
            df = pd.read_csv(log_path)
        except Exception as e:
            print(f"[Inference] Warning: Cannot read existing result {log_path}: {e}")
            return False
        required_columns = {"timestamp", "rppg"}
        if df.empty or not required_columns.issubset(df.columns):
            return False

        timestamp = pd.to_numeric(df["timestamp"], errors="coerce")
        rppg = pd.to_numeric(df["rppg"], errors="coerce")
        valid_rows = np.isfinite(timestamp) & np.isfinite(rppg)
        return bool(valid_rows.all())

    def _log_results(self, path=None):
        timestamps = []
        results = []
        while not self.result_queue.empty():
            result, timestamp = self.result_queue.get()
            timestamps.append(timestamp)
            results.append(result)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "rppg": results
        })

        df.to_csv(os.path.join(path, "rppg_log.csv"), index=False)

    def _inference(self, path=None):
        if os.path.isdir(path) is False:
            print(f"[Inference] Error: {path} is not a valid directory.")
            return

        inference_vars.inference_completed = False
        inference_vars.preprocess_completed = False
        self._ensure_model()

        self.video2frame = v2f.Video2Frame(path)

        threads = []
        preprocess_thread = threading.Thread(target=self.video2frame, args=(self.preprocess_queue,))
        model_thread = threading.Thread(target=self.model, args=(self.preprocess_queue, self.result_queue))
        threads.append(preprocess_thread)
        threads.append(model_thread)

        for thread in threads:
            thread.start()

        while not (inference_vars.inference_completed and inference_vars.preprocess_completed):
            time.sleep(0.01)

        for thread in threads:
            thread.join(timeout=5)
            if thread.is_alive():
                print(f"[Inference] Warning: Thread {thread.name} did not terminate in time.")

        self._log_results(path)

    def __call__(self, starting_point=None, ending_point=None):
        if os.path.isdir(self.data_dir) is False:
            print(f"[Inference] Error: {self.data_dir} is not a valid directory.")
            return

        dirs = []
        for dir in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, dir)):
                dir_index = int(dir[8:]) if dir[8:].isdigit() else -1
                if starting_point is not None and dir_index < starting_point:
                    continue
                if ending_point is not None and dir_index > ending_point:
                    continue
                dirs.append(dir)

        for dir in tqdm(dirs):
            path = os.path.join(self.data_dir, dir)
            if self.skip_existing and self._has_existing_results(path):
                print(f"[Inference] Skipping already inferenced directory: {dir}")
                continue
            print(f"[Inference] Processing directory: {dir}")
            self._inference(path=path)
            if global_vars.user_interrupt:
                break

def main():
    signal.signal(signal.SIGINT, signal_handler)
    path = input("Input inference path:").strip()
    mirror_version = input("Input mirror version (1 or 2, default 1):").strip() or "1"
    if mirror_version not in {"1", "2"}:
        print(f"[Inference] Error: {mirror_version} is not a valid mirror version.")
        return
    global_vars.mirror_version = mirror_version
    skip_existing_input = input("Skip already inferenced folders? (Y/n, default y):").strip().lower()
    if skip_existing_input in {"", "y", "yes", "1", "true"}:
        skip_existing = True
    elif skip_existing_input in {"n", "no", "0", "false"}:
        skip_existing = False
    else:
        print(f"[Inference] Error: {skip_existing_input} is not a valid skip option.")
        return
    starting_point = input("Input starting point (default no limit):").strip()
    ending_point = input("Input ending point (default no limit):").strip()
    
    start = int(starting_point) if starting_point.isdigit() else None
    end = int(ending_point) if ending_point.isdigit() else None

    local_inference = LocalInference(data_dir=path, mirror_version=mirror_version, skip_existing=skip_existing)
    local_inference(starting_point=start, ending_point=end)


if __name__ == "__main__":
    main()
