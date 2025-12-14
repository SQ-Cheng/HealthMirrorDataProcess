import global_vars
from queue import Queue
from model.step import Step
import preprocess.video2frame as v2f
from log.result_log import ResultLogger
from log.oldmerge import OldFileMerger
from log.merge import FileMerger

from log.normalize import Normalizer
import threading
import time

import os
import sys
import signal

mirror_version = global_vars.mirror_version

def signal_handler(sig, frame):
    global_vars.user_interrupt = True

def inference(model_choice="Step", path=None):
    if os.path.isdir(path) is False:
        print(f"Error: {path} is not a valid directory.")
        return
    
    model_choice = "Step"
    preprocess_queue = Queue()
    result_queue = Queue()
    video2frame = v2f.Video2Frame()
    
    if model_choice == "Step":
        model = Step(
            model_path="./model/models/onnx/step.onnx",
            state_path="./model/models/onnx/state.pkl",
            dt=1 / 30
        )

    global_vars.user_interrupt = False
    
    video2frame.path = path
    result_logger = ResultLogger(path + "/rppg_log.csv")
    if mirror_version == "1":
        file_merger = OldFileMerger([path + "/rppg_log.csv", path + "/ecg_log.csv"], path + "/merged_log.csv")
        normalizer = Normalizer(path + "/merged_log.csv", path + "/normalized_log.csv")
    elif mirror_version == "2":
        file_merger = FileMerger([(["rppg"], path + "/rppg_log.csv"), (["ppg_red","ppg_ir","ppg_green","ecg"], path + "/merged_log.csv")], path + "/merged_output.csv")
        normalizer = Normalizer(path + "/merged_output.csv", path + "/normalized_log.csv")

    threads = []

    preprocess_thread = threading.Thread(target=video2frame, args=(preprocess_queue,))
    model_thread = threading.Thread(target=model, args=(preprocess_queue, result_queue))

    threads.append(preprocess_thread)
    threads.append(model_thread)

    for thread in threads:
        thread.start()

    try:
        message_printed = False
        while not (global_vars.inference_completed and global_vars.preprocess_completed):
            if global_vars.user_interrupt:
                break
            time.sleep(1)
            if not message_printed:
                print("Preprocessed frames:", video2frame.processed_frames)
                message_printed = True
    except KeyboardInterrupt:
        global_vars.user_interrupt = True

    for thread in threads:
        thread.join(timeout=5)
        if thread.is_alive():
            print(f"Warning: Thread {thread.name} did not terminate in time.")

    if not global_vars.user_interrupt:
        result_logger.log(result_queue)
        time.sleep(0.5)
        file_merger()
        time.sleep(0.5)
        normalizer()
        print("Inference on {} completed. Results logged.".format(path))

    else:
        print("Inference was interrupted")

def inference_handler(path, dir):
    with open(os.path.join(path, dir, "video.avi.ts"), 'r') as f:
        f.readline()
        video_begin_time = float(f.readline().strip().split(',')[1])
    with open(os.path.join(path, dir, "ecg_log.csv"), 'r') as f:
        f.readline()
        ecg_begin_time = float(f.readline().strip().split(',')[0])
    if abs(video_begin_time - ecg_begin_time) > 3.0:
        print(f"Warning: Time difference between video and ECG for {dir} is greater than 3 seconds.")
        with open("time_diff_warning.txt", 'a') as f:
            f.write(f"path: {os.path.join(path, dir)}, time difference: {abs(video_begin_time - ecg_begin_time)} seconds\n")
    inference(path=os.path.join(path, dir))

def main():
    signal.signal(signal.SIGINT, signal_handler)
    path = input("Input inference path:").strip()
    starting_point = input("Input starting point (default 0, -1 for smart inference):").strip()
    if starting_point.isdigit() and int(starting_point) > 0:
        for dir in os.listdir(path):
            if os.path.isdir(os.path.join(path, dir)):
                if not os.path.exists(os.path.join(path, dir, "rppg_log.csv")):
                    open(os.path.join(path, dir, "rppg_log.csv"), 'w')
                if int(dir[8:]) < int(starting_point):
                    continue
                inference_handler(path, dir)
                if global_vars.user_interrupt:
                    break

    elif starting_point == "-1":
        for dir in os.listdir(path):
            if os.path.isdir(os.path.join(path, dir)):
                if not os.path.exists(os.path.join(path, dir, "rppg_log.csv")):
                    open(os.path.join(path, dir, "rppg_log.csv"), 'w')
                with open (os.path.join(path, dir, "rppg_log.csv"), 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        print(f"Skipping {dir}: already processed.")
                        continue
                inference_handler(path, dir)
                if global_vars.user_interrupt:
                    break

if __name__ == "__main__":
    main()