import os
import pandas as pd
import global_vars

class DataLoader:
    def __init__(self, raw_dir=None, cleaned_dir=None):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.mirror_version = global_vars.mirror_version

    def load_raw_data(self, patient_id=[]):
        for f in os.listdir(self.raw_dir):
            if int(f[8:]) in patient_id or not patient_id:
                try:
                    file_path = os.path.join(self.raw_dir, f, "merged_log.csv")
                    df = pd.read_csv(file_path)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                yield int(f[8:]), df

    def load_cleaned_data(self, patient_id=[]):
        for f in os.listdir(self.cleaned_dir):
            if f.endswith(".csv") and (int(f[8:14]) in patient_id or not patient_id):
                try:
                    file_path = os.path.join(self.cleaned_dir, f)
                    df = pd.read_csv(file_path)
                    
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

                yield int(f[8:14]), df