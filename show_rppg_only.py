import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_overall_patient_info(csv_file='overall_patient_info.csv'):
    try:
        df = pd.read_csv(csv_file, dtype={'hospital_patient_id': str})
        return df
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return None

def search_by_hospital_id(hospital_id):
    df = load_overall_patient_info()
    if df is not None:
        try:
            valid_df = df[df['hospital_patient_id'] != 'n/a']
            print(f"Availiable hospital_patient_ids: {valid_df['hospital_patient_id'].unique()}")
            result = valid_df[valid_df['hospital_patient_id'].astype(str) == str(hospital_id)]
            
        except Exception as e:
            print(f"Error searching for hospital_patient_id {hospital_id}: {e}")
            return []
        if not result.empty:
            return result['lab_patient_id'].tolist()
        else:
            print(f"No records found for hospital_patient_id: {hospital_id}")
            return []
    else:
        return []
    
def plot_rppg_signal(timestamps, rppg_signal, title="rPPG Signal"):
    plt.figure(figsize=(18, 6))
    plt.plot(timestamps, rppg_signal, label='rPPG Signal', color='blue')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('rPPG Amplitude')
    plt.legend()
    plt.grid()
    plt.show()


def plot_by_hospital_id(hospital_id):
    lab_ids = search_by_hospital_id(hospital_id)
    if not lab_ids:
        print(f"No lab_patient_ids found for hospital_patient_id: {hospital_id}")
        return
    
    for lab_id in lab_ids:
        rppg_file = f"./patient_data/patient_{lab_id:0>6}/rppg_log.csv"
        try:
            # timestamps in the first column, rPPG signal in the second column
            timestamps = pd.read_csv(rppg_file).iloc[:, 0].values
            rppg_signal = pd.read_csv(rppg_file).iloc[:, 1].values
            plot_rppg_signal(timestamps, rppg_signal, title=f"rPPG Signal for Lab Patient ID: {lab_id}")
        except Exception as e:
            print(f"Error loading rPPG data for lab_patient_id {lab_id}: {e}")
            continue

def main():
    hospital_id = input("Enter hospital_patient_id to plot rPPG signals: ")
    plot_by_hospital_id(hospital_id)

if __name__ == "__main__":
    main()