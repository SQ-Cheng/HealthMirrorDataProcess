from .base import LogBase
from queue import Queue
import os
import global_vars

class ResultLogger(LogBase):
    def __init__(self, log_file='results.csv'):
        self.log_file = log_file

    def log(self, results_queue: Queue):
        logged_lines = 0
        with open(self.log_file, 'w') as f:
            if global_vars.mirror_version != "1":
                f.write('timestamp,rppg\n')
            while not results_queue.empty():
                result, timestamp = results_queue.get()
                f.write(f'{timestamp},{result[0]}\n')
                logged_lines += 1
        f.close()
        print(f"[ResultLogger] {logged_lines} result datapoints logged to {self.log_file}")
