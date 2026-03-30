import cv2
import numpy as np
from queue import Queue
from .base import PreprocessBase
import inference_vars


class Video2Frame(PreprocessBase):
    def __init__(self, path=None, log=False):
        self.path = path
        self.video_path = None
        self.ts_path = None
        self.cap = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        self.processed_frames = 0
        inference_vars.preprocess_completed = False
        self.log = log

    def _preprocess_conv_format(self, frame) -> np.ndarray:
        frame = cv2.resize(frame, (36, 36))
        return frame#.astype("float32")

    def _load_timestamps(self):
        timestamps = []
        try:
            with open(self.ts_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        timestamps.append(float(parts[1]))
        except Exception as e:
            print(f"[Preprocess] Error reading timestamp file: {e}")
            return []
        return timestamps

    def __call__(self, preprocess_queue: Queue):
        self.video_path = self.path + "/video.avi"
        self.ts_path = self.path + "/video.avi.ts"

        timestamps = self._load_timestamps()
        
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"[Preprocess] Error: Cannot open video file {self.video_path}")
            self.cap.release()
            inference_vars.preprocess_completed = True
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.log:
            print(f"Loaded {len(timestamps)} timestamps")

        self.processed_frames = 0
        while True:
            ret, raw_frame = self.cap.read()
            if not ret:
                ret, raw_frame = self.cap.read()
                if not ret:
                    break

            timestamp = timestamps[self.processed_frames]
                
            preprocess_queue.put((self._preprocess_conv_format(raw_frame), timestamp))
            self.processed_frames += 1

        self.cap.release()
        print(f"[Preprocess] Processed {self.processed_frames}/{len(timestamps)} frames.")
        inference_vars.preprocess_completed = True
