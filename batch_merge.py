from log.merge_new import FileMerger
import matplotlib.pyplot as plt
import global_vars
import os
import gc

global_vars.mirror_version = "1"
"""
merge1 = FileMerger(path="./mirror1_data/patient_000528", log=True)
df1 = merge1()
while True:
    pass
"""
negative = True

def main():
    global_vars.mirror_version = "1"
    data_dir = input("Enter mirror directory path: ")
    
    for dir in os.listdir(data_dir):
        if dir.startswith("patient_"):
            try:
                patient_path = os.path.join(data_dir, dir)
                merger = FileMerger(path=patient_path, log=True, negative=negative)
                merged_df = merger()
                if merged_df is None:
                    print(f"Skipping {patient_path} due to merge issues.")
                    continue
                print(f"Merged data for {patient_path}:")
                print(merged_df.head())
                del merger, merged_df
                gc.collect()
            except Exception as e:
                print(f"Error processing {patient_path}: {e}")
                

if __name__ == "__main__":
    main()