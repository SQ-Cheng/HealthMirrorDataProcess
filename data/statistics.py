from load_data import DataLoader
from patient_info import PatientInfo
import numpy as np
import scipy.signal as signal
import os
import pandas as pd

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def find_peaks(data, threshold=-0.2, fs=512):
    peaks, _ = signal.find_peaks(data, height=threshold, distance=fs*0.5)
    return peaks

def calculate_hr(peaks, timestamps):
    if len(peaks) < 2:
        return None
    rr_intervals = np.diff(timestamps[peaks])
    hr = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else None
    return hr

class PatientStatistics:
    def __init__(self, raw_data_dir, cleaned_data_dir):
        self.raw_data_dir = raw_data_dir
        self.cleaned_data_dir = cleaned_data_dir
        self.patient_list = self.load_all_patients()

    def load_all_patients(self, data_dir="./patient_data", output_file="overall_patient_info.csv"):
        patient_info = PatientInfo(data_dir, save_dir=output_file, mode="dir")
        patient_info_list = patient_info.extract()
        return patient_info_list
    
    def load_data_for_patients(self, patient_list, raw_dir="./patient_data", cleaned_dir="./test_cleaned"):
        patient_ids = [int(p['lab_patient_id']) for p in patient_list]
        data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
        raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
        cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
        return raw_data, cleaned_data
    
    def count_raw_data_points(self, start_idx=0, end_idx=None):
        total_points = 0
        for f in os.listdir(self.raw_data_dir):
            try:
                if int(f[8:]) >= start_idx and (end_idx is None or int(f[8:]) < end_idx):
                    file_path = os.path.join(self.raw_data_dir, f, "merged_log.csv")
                    df = pd.read_csv(file_path)
                    total_points += len(df)
            except Exception as e:
                continue
        print(f"原始数据：{start_idx} 到 {end_idx}: {total_points}, {total_points/(60*512):.2f}分钟")
        return total_points

    def count_cleaned_data_points(self, start_idx=0, end_idx=None):
        total_points = 0
        for f in os.listdir(self.cleaned_data_dir):
            try:
                if f.endswith(".csv") and int(f[8:14]) >= start_idx and (end_idx is None or int(f[8:14]) < end_idx):
                    file_path = os.path.join(self.cleaned_data_dir, f)
                    df = pd.read_csv(file_path)
                    total_points += len(df)
            except Exception as e:
                continue
        print(f"清洗数据：{start_idx} 到 {end_idx}: {total_points}, {total_points/(60*512):.2f}分钟")
        return total_points
        
    def overall_stat(self):
        print(f"人次: {len(self.patient_list)}")
        hospital_ids = set(p['hospital_patient_id'] for p in self.patient_list if p['hospital_patient_id'] != 'n/a')
        print(f"人数: {len(hospital_ids)}")

    def hr_stat(self):
        raw_data, cleaned_data = self.load_data_for_patients(self.patient_list)
        rppg_hrs = []
        ecg_hrs = []
        for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
            rppg_signal = filter_signal(rppg_signal, fs=512)
            ecg_signal = filter_signal(ecg_signal, fs=512)
            rppg_peaks = find_peaks(rppg_signal, fs=512)
            ecg_peaks = find_peaks(ecg_signal, fs=512)
            rppg_hr = calculate_hr(rppg_peaks, timestamps)
            ecg_hr = calculate_hr(ecg_peaks, timestamps)
            rppg_hrs.append(rppg_hr)
            ecg_hrs.append(ecg_hr)

        print(f"rPPG心率均值: {np.nanmean(rppg_hrs):.2f}, ECG心率均值: {np.nanmean(ecg_hrs):.2f}")
        print(f"rPPG心率最大值: {np.nanmax(rppg_hrs):.2f}, ECG心率最大值: {np.nanmax(ecg_hrs):.2f}")
        print(f"rPPG心率最小值: {np.nanmin(rppg_hrs):.2f}, ECG心率最小值: {np.nanmin(ecg_hrs):.2f}")

    def bp_stat(self):
        patient_with_bp = [p for p in self.patient_list if p['low_blood_pressure'] != -1 and p['high_blood_pressure'] != -1]
        low_bps = [p['low_blood_pressure'] for p in patient_with_bp]
        high_bps = [p['high_blood_pressure'] for p in patient_with_bp]
        print(f"血压均值：{np.mean(low_bps):.2f}/{np.mean(high_bps):.2f}")
        print(f"血压最大值：{np.max(low_bps)}/{np.max(high_bps)}")
        print(f"血压最小值：{np.min(low_bps)}/{np.min(high_bps)}")
    
def main():
    stats = PatientStatistics(raw_data_dir="./mirror2_data", cleaned_data_dir="./mirror2_cleaned")
    """
    stats.hr_stat()
    stats.bp_stat()
    stats.count_raw_data_points()
    stats.count_cleaned_data_points()
    stats.overall_stat()
    """
    start_idx = 122
    stats.count_raw_data_points(start_idx=start_idx)
    stats.count_cleaned_data_points(start_idx=start_idx)

if __name__ == "__main__":
    main()
