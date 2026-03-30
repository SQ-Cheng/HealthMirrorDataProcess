from abc import abstractmethod
from queue import Queue


class LogBase:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, results_queue: Queue):
        pass
