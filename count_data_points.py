import os
import pandas as pd
def count_raw_data_points(data_dir = "./patient_data", start_idx=0, end_idx=None):
    total_points = 0
    for f in os.listdir(data_dir):
        if int(f[8:]) >= start_idx and (end_idx is None or int(f[8:]) < end_idx):
            file_path = os.path.join(data_dir, f, "merged_log.csv")
            df = pd.read_csv(file_path)
            total_points += len(df)
    print(f"TRaw: from {start_idx} to {end_idx}: {total_points}, {total_points/(60*512):.2f} minutes")
    return total_points

def count_cleaned_data_points(data_dir = "./test_cleaned", start_idx=0, end_idx=None):
    total_points = 0
    for f in os.listdir(data_dir):
        if f.endswith(".csv") and int(f[8:14]) >= start_idx and (end_idx is None or int(f[8:14]) < end_idx):
            file_path = os.path.join(data_dir, f)
            df = pd.read_csv(file_path)
            total_points += len(df)
    print(f"Cleaned: from {start_idx} to {end_idx}: {total_points}, {total_points/(60*512):.2f} minutes")
    return total_points

idx = 255
count_raw_data_points(start_idx=idx)
count_cleaned_data_points(start_idx=idx)