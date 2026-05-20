from queue import Queue
from queue import Empty
import pickle
import onnxruntime as ort
import numpy as np

import global_vars
from .base import ModelBase
import inference_vars


class Step(ModelBase):
    def __init__(self, model_path, state_path, dt=None):
        super().__init__()
        self.model_path = model_path
        self.state_path = state_path
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. "
                "Please install/activate onnxruntime-gpu for GPU inference."
            )
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in self.model.get_providers():
            raise RuntimeError(
                "CUDAExecutionProvider failed to initialize. "
                "Please ensure CUDA 12.x and cuDNN 9.x libraries are loadable."
            )
        with open(state_path, "rb") as f:
            self.state = pickle.load(f)
        self.dt = np.array(dt).astype("float16")
        self.last_timestamp = None
        inference_vars.inference_completed = False

    def __call__(self, preprocess_queue: Queue, result_queue: Queue):
        while not inference_vars.inference_completed and not global_vars.user_interrupt:
            try:
                frame, timestamp = preprocess_queue.get(timeout=1)
                if self.last_timestamp is None:
                    self.last_timestamp = timestamp
                    dt = self.dt
                else:
                    dt = timestamp - self.last_timestamp
                    dt = np.array(dt).astype("float16")
                    self.last_timestamp = timestamp
                    
            except Empty:
                if inference_vars.preprocess_completed:
                    inference_vars.inference_completed = True
                break
                
            image = np.array([[frame]]).astype("float16") / 255.0
            input_dict = {"arg_0.1": image, "onnx::Mul_37": dt, **self.state}
            result = self.model.run(None, input_dict)
            self.state = dict(zip(list(input_dict)[2:], result[1:]))
            result_queue.put((result[0][0, 0], timestamp))
        with open(self.state_path, "wb") as f:
            pickle.dump(self.state, f)

        self.last_timestamp = None
            
        inference_vars.inference_completed = True
