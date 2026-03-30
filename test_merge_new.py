from log.merge_new import FileMerger
import matplotlib.pyplot as plt
import global_vars
import os

global_vars.mirror_version = "1"
merge1 = FileMerger(path="./mirror1_data/patient_000002", log=True)
df1 = merge1()

def main():
    global_vars.mirror_version = "1"
    data_dir = input("Enter data directory path: ")
    for dir in os.listdir(data_dir):
        if dir.startswith("patient_"):
            try:
                patient_path = os.path.join(data_dir, dir)
                merger = FileMerger(path=patient_path, log=True)
                merged_df = merger()
                print(f"Merged data for {patient_path}:")
                print(merged_df.head())
            except Exception as e:
                print(f"Error processing {patient_path}: {e}")