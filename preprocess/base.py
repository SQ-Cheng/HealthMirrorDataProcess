from abc import abstractmethod
from queue import Queue


class PreprocessBase:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, preprocess_queue: Queue):
        pass
