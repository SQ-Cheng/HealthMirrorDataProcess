import os
import pandas as pd
import numpy as np
from data.patient_info import PatientInfo
from data.load_data import DataLoader
from ecg.ecg_process import ECGProcess
from utils.signal_processing import calculate_ptt

# Configuration
lab = False
if lab:
    data_dir = "./lab_mirror_data"
    output_file = "lab_overall_patient_info.csv"
    merged_patient_file = "lab_merged_patient_info.csv"
    cleaned_dir = "./lab_test_cleaned"
else:
    data_dir = "./mirror1_data"
    output_file = "overall_patient_info.csv"
    merged_patient_file = "merged_patient_info.csv"
    cleaned_dir = "./test_sliced"

def load_patient_with_bp(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    patient_with_bp = [p for p in patient_info_list if int(p['low_blood_pressure']) != -1 and int(p['high_blood_pressure']) != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir=data_dir, cleaned_dir=cleaned_dir):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    cleaned_data_loader = data_loader.load_cleaned_data(patient_id=patient_ids)
    return cleaned_data_loader

def main():
    bp_patient_list = load_patient_with_bp()
    cleaned_data_loader = load_data_for_patients(bp_patient_list)

    # Use Pan-Tompkins method as requested
    ecg_processor = ECGProcess(method='pt', fs=512)

    results = []

    print("Starting PTT extraction...")

    for patient_id, df in cleaned_data_loader:
        if df is None:
            continue

        print(f"Processing patient {patient_id}...")

        timestamps = None
        for col in ['Timestamp', 'Time', 'timestamp', 'timestamps']:
            if col in df.columns:
                timestamps = df[col].to_numpy()
                break

        ecg_signal = None
        for col in ['ECG', 'ecg']:
            if col in df.columns:
                ecg_signal = df[col].to_numpy()
                break

        rppg_signal = None
        for col in ['RPPG', 'rppg', 'rPPG']:
            if col in df.columns:
                rppg_signal = df[col].to_numpy()
                break

        if timestamps is None or ecg_signal is None or rppg_signal is None:
            print(f"Missing data for patient {patient_id}")
            continue

        ecg_processor.process(ecg_signal)
        ecg_peaks = ecg_processor.get_peaks()
        if ecg_peaks is None or len(ecg_peaks) == 0:
            print(f"Patient {patient_id}: Could not detect ECG peaks")
            continue

        ptt, std = calculate_ptt(timestamps, rppg_signal, ecg_signal, ecg_peaks)

        if ptt is not None:
            results.append({
                'patient_id': patient_id,
                'ptt': ptt,
                'ptt_std': std
            })
            print(f"Patient {patient_id}: PTT = {ptt:.4f}, Std = {std:.4f}")
        else:
            print(f"Patient {patient_id}: Could not calculate PTT")

    output_csv = "ptt_bp.csv"
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
