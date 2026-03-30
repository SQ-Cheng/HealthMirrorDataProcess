import global_vars
import inference_vars
from queue import Queue
from model.step import Step
import preprocess.video2frame as v2f

import threading
import time
import pandas as pd
import os
import signal
from tqdm import tqdm

mirror_version = global_vars.mirror_version

def signal_handler(sig, frame):
    global_vars.user_interrupt = True

class LocalInference:
    def __init__(self, model_choice="Step", mirror_version="1", data_dir=None):
        self.model_choice = model_choice
        self.mirror_version = mirror_version
        self.data_dir = data_dir
        self.model = self._init_model()
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
            print(f"[Inference] Processing directory: {dir}")
            self._inference(path=os.path.join(self.data_dir, dir))
            if global_vars.user_interrupt:
                break

def main():
    signal.signal(signal.SIGINT, signal_handler)
    path = input("Input inference path:").strip()
    starting_point = input("Input starting point (default no limit):").strip()
    ending_point = input("Input ending point (default no limit):").strip()
    
    start = int(starting_point) if starting_point.isdigit() else None
    end = int(ending_point) if ending_point.isdigit() else None

    local_inference = LocalInference(data_dir=path)
    local_inference(starting_point=start, ending_point=end)


if __name__ == "__main__":
    main()