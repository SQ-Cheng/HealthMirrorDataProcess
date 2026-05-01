from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import spearmanr
from utils.signal_processing import filter_signal

lab = False

if lab:
    data_dir = "./lab_mirror_data"
    output_file = "lab_overall_patient_info.csv"
    merged_patient_file = "lab_merged_patient_info.csv"
    cleaned_dir = "./lab_test_cleaned"
else:
    data_dir = "./patient_data"
    output_file = "overall_patient_info.csv"
    merged_patient_file = "merged_patient_info.csv"
    cleaned_dir = "./test_sliced"

def load_all_patients(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    return patient_info_list

def load_patient_with_bp(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    # Convert string values to int for proper comparison
    patient_with_bp = [p for p in patient_info_list if int(p['low_blood_pressure']) != -1 and int(p['high_blood_pressure']) != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir=data_dir, cleaned_dir=cleaned_dir):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    raw_data = list(data_loader.load_raw_data(patient_id=patient_ids))
    cleaned_data = list(data_loader.load_cleaned_data(patient_id=patient_ids))
    return raw_data, cleaned_data

def ecg_peak_pantompkins(ecg_signal, fs=512):
    # Bandpass filter
    filtered_ecg = filter_signal(ecg_signal, fs=fs, lowcut=5.0, highcut=15.0, order=2)
    
    # Derivative to enhance QRS complexes
    derivative_ecg = np.gradient(filtered_ecg)
    
    # Squaring
    squared_ecg = np.square(derivative_ecg)
    
    # Moving average
    window_size = int(0.150 * fs)  # 150 ms window
    pantompkins = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
    pantompkins = 2 * pantompkins / np.max(pantompkins)
    print(pantompkins)
    
    # Peak detection with prominence and width constraints
    signal_std = np.std(pantompkins)
    max_width = int(fs * 0.5)  # Max width ~120 ms

    pt_peaks, _ = signal.find_peaks(
        pantompkins,
        distance=int(fs * 0.35),
        width=(1, max_width)
    )

    peaks = []
    for peak in pt_peaks:
        left = max(0, peak - int(0.1 * fs))
        right = min(len(ecg_signal), peak + int(0.1 * fs))
        p = np.argmax(ecg_signal[left:right]) + left
        peaks.append(p)
    
    print (f"Detected {len(peaks)}/{len(pt_peaks)} ECG peaks using Pan-Tompkins algorithm.")
        
    return peaks, pantompkins

def find_peaks_new(signal_data, signal_type='rppg', fs=512, min_distance=None):
    """
    在信号中寻找峰值，针对不同信号类型使用不同策略
    (从data_slicer.py移植)
    
    参数:
        signal_data: 输入信号
        signal_type: 信号类型 ('rppg' 或 'ecg')
        fs: 采样频率
        min_distance: 峰值之间的最小距离（样本点数），默认为0.5秒
    """
    if min_distance is None:
        min_distance = int(fs * 0.35)
    
    if signal_type == 'ecg':
        peaks, pantompkins = ecg_peak_pantompkins(signal_data, fs=fs)
        return peaks, pantompkins
    else:
        peaks, properties = signal.find_peaks(signal_data, distance=min_distance, height=0)
        return peaks, properties
    
    

def calculate_ptt_new(time, rppg_signal, ecg_signal):
    """
    估算PTT值 (从data_slicer.py移植)
    
    参数:
        time: 时间戳数组
        rppg_signal: rPPG信号
        ecg_signal: ECG信号
        
    返回:
        ptt: 估算的PTT值（秒），如果无法估算则返回None
        rppg_peaks: rPPG峰值索引
        ecg_peaks: ECG峰值索引
        std: PTT值的标准差
    """
    # 寻找峰值
    rppg_peaks, _ = find_peaks_new(rppg_signal, signal_type='rppg', fs=512)
    ecg_peaks, _ = find_peaks_new(ecg_signal, signal_type='ecg', fs=512)
    
    if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
        return None, None, None
    
    # 为每个ECG峰找到最近的后续rPPG峰
    matched_pairs = []
    ptt_values = []
    
    for ecg_idx in ecg_peaks:
        ecg_time = time[ecg_idx]
        
        # 找到在ECG峰之后的rPPG峰
        future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]
        
        if len(future_rppg_peaks) > 0:
            # 选择最近的rPPG峰
            rppg_idx = future_rppg_peaks[0]
            rppg_time = time[rppg_idx]
            
            # 计算PTT（应该为正值，因为rPPG在ECG之后）
            ptt = rppg_time - ecg_time

            # 只接受合理范围内的PTT (0.05s到0.4s，即50ms到400ms)
            if 0.05 < ptt < 0.4:
                matched_pairs.append((ecg_idx, rppg_idx))
                ptt_values.append(ptt)
    
    if len(ptt_values) == 0:
        return None, None, None
    
    # 使用中位数来避免异常值的影响
    ptt_median = np.median(ptt_values)
    
    # 过滤掉偏差过大的值
    ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
    
    if len(ptt_filtered) == 0:
        return None, None, None
    
    ptt_final = np.mean(ptt_filtered)
    std = np.std(ptt_filtered)
    
    return ptt_final, None, std

def ptt_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        rppg_signal = filter_signal(rppg_signal, fs=fs)
        ecg_signal = filter_signal(ecg_signal, fs=fs)
    if peak:
        ptt, align, std = calculate_ptt_new(timestamps, rppg_signal, ecg_signal)
        if ptt is not None:
            print(f"PTT: {ptt:.3f} seconds, Std: {std:.3f}")
        return ptt, align, std
    return None, None, None

def plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512):
    if filter:
        filter_signal(rppg_signal, fs=fs)
        filter_signal(ecg_signal, fs=fs)
    if peak:
        ptt, align, std = calculate_ptt_new(timestamps, rppg_signal, ecg_signal)
        if ptt is not None:
            print(f"PTT: {ptt:.3f} seconds, Std: {std:.3f}")
        
        # 重新计算峰值用于显示
        rppg_peaks, _ = find_peaks_new(rppg_signal, signal_type='rppg', fs=fs)
        ecg_peaks, pantompkins = find_peaks_new(ecg_signal, signal_type='ecg', fs=fs)
        print(f"ecg: {len(ecg_signal)}, pan-tompkins: {len(pantompkins)}, rppg: {len(rppg_signal)}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, rppg_signal, label='rPPG Signal')
    if peak:
        plt.plot(timestamps[rppg_peaks], rppg_signal[rppg_peaks], "x", label='rPPG Peaks')
    plt.title('rPPG Signal with Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, ecg_signal, label='ECG Signal')
    plt.plot(timestamps, pantompkins, label='Pan-Tompkins')
    if peak:
        plt.plot(timestamps[rppg_peaks], rppg_signal[rppg_peaks], "x", label='rPPG Peaks')
        plt.plot(timestamps[ecg_peaks], ecg_signal[ecg_peaks], "o", label='ECG Peaks')
    plt.title('ECG Signal with Peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
    option = input(f"STD: {std}, Y to accept, N to reject: ")
    if option.lower() == 'n':
        return None, None, None
    return ptt, align, std


def show_patient_with_hr():
    patient_list = load_patient_with_bp()
    raw_data, cleaned_data = load_data_for_patients(patient_list)

    for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
        print(f"Patient ID: {patient_id}")
        plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)

bp_patient_list = load_patient_with_bp()
raw_data, cleaned_data = load_data_for_patients(bp_patient_list)

bp_dict = {int(p['lab_patient_id']): (int(p['low_blood_pressure']), int(p['high_blood_pressure'])) 
           for p in bp_patient_list}

ptt_list = []
for patient_id, timestamps, rppg_signal, ecg_signal in cleaned_data:
    print(f"Patient ID: {patient_id}")
    ptt, align, std = plot_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)
    #ptt, align, std = ptt_signals(timestamps, rppg_signal, ecg_signal, peak=True, filter=True, fs=512)
    ptt_list.append((patient_id, ptt, align, std))

best_coef = 0
best_std = 0
best_ptt_low_threshold = 0
for std_threshold in np.linspace(0.005, 0.2, 40):
    for ptt_low_threshold in np.linspace(0, 0.2, 21):
        ptt_bp = []
        for patient_id, ptt, align, std in ptt_list:
            if std is None or std > std_threshold:
                continue
            if patient_id in bp_dict:
                low_bp, high_bp = bp_dict[patient_id]
                if(ptt != None and ptt_low_threshold < abs(ptt) and abs(ptt) < 1 and low_bp != -1 and high_bp != -1):
                    ptt_bp.append((patient_id, ptt, low_bp, high_bp, std))

        # calculate the average ptt for patients
        ptt_dict = {}
        for patient_id, ptt, low_bp, high_bp, std in ptt_bp:
            if ptt is not None:
                if patient_id not in ptt_dict:
                    ptt_dict[patient_id] = []
                ptt_dict[patient_id].append((ptt, low_bp, high_bp, std))
        ptt_bp = [(np.mean([p[0] for p in v]), v[0][1], v[0][2], np.mean([p[3] for p in v])) for k, v in ptt_dict.items()]
        ptt_bp.sort(key=lambda x: x[3])  # sort by std

        ptt_values = []
        low_bps = []
        high_bps = []
        mean_bps = []
        for i in range(len(ptt_bp)):
            ptt_values.append(ptt_bp[i][0])
            high_bps.append(ptt_bp[i][2])
            low_bps.append(ptt_bp[i][1])
            mean_bps.append((ptt_bp[i][1]+ptt_bp[i][2])/2)

        # minimum 15 data points to calculate correlation
        if len(ptt_values) < 8:
            continue

        ptt_values = np.array(ptt_values)
        ptt_values_rec = np.reciprocal(ptt_values)
        low_bps = np.array(low_bps)
        high_bps = np.array(high_bps)
        mean_bps = np.array(mean_bps)

        low_coef = np.corrcoef(ptt_values, low_bps)[0, 1]
        low_coef_rec = np.corrcoef(ptt_values_rec, low_bps)[0, 1]
        high_coef = np.corrcoef(ptt_values, high_bps)[0, 1]
        high_coef_rec = np.corrcoef(ptt_values_rec, high_bps)[0, 1]
        mean_coef = np.corrcoef(ptt_values, mean_bps)[0, 1]
        mean_coef_rec = np.corrcoef(ptt_values_rec, mean_bps)[0, 1]
        coef = low_coef + high_coef + mean_coef
        if coef > best_coef and low_coef > 0 and high_coef > 0 and mean_coef > 0:
            best_coef = coef
            best_std = std_threshold
            best_ptt_low_threshold = ptt_low_threshold
            print(f"New best coef: {best_coef:.2f} with std threshold: {best_std}, ptt low threshold: {best_ptt_low_threshold}")
            print(f"Low coef: {low_coef:.2f}, High coef: {high_coef:.2f}, Mean coef: {mean_coef:.2f}")
            print(f"Low coef rec: {low_coef_rec:.2f}, High coef rec: {high_coef_rec:.2f}, Mean coef rec: {mean_coef_rec:.2f}")
print(f"Best coef: {best_coef:.2f} with std threshold: {best_std}, ptt low threshold: {best_ptt_low_threshold}")

ptt_bp = []
for patient_id, ptt, align, std in ptt_list:
    if std is None or std > best_coef:
        continue
    if patient_id in bp_dict:
        low_bp, high_bp = bp_dict[patient_id]
        if(ptt != None and best_ptt_low_threshold < abs(ptt) and abs(ptt) < 1 and low_bp != -1 and high_bp != -1):
            ptt_bp.append((patient_id, ptt, low_bp, high_bp, std))
            print(f"Patient ID: {patient_id}, PTT: {ptt:.3f} seconds, Low BP: {low_bp}, High BP: {high_bp}, Std: {std:.3f}")

# calculate the average ptt for patients
ptt_dict = {}
for patient_id, ptt, low_bp, high_bp, std in ptt_bp:
    if ptt is not None:
        if patient_id not in ptt_dict:
            ptt_dict[patient_id] = []
        ptt_dict[patient_id].append((ptt, low_bp, high_bp, std))
ptt_bp = [(np.mean([p[0] for p in v]), v[0][1], v[0][2], np.mean([p[3] for p in v])) for k, v in ptt_dict.items()]
ptt_bp.sort(key=lambda x: x[3])  # sort by std

ptt_values = []
low_bps = []
high_bps = []
mean_bps = []
for i in range(len(ptt_bp)):
    ptt_values.append(ptt_bp[i][0])
    high_bps.append(ptt_bp[i][2])
    low_bps.append(ptt_bp[i][1])
    mean_bps.append((ptt_bp[i][1]+ptt_bp[i][2])/2)

ptt_values = np.array(ptt_values)
ptt_values_rec = np.reciprocal(ptt_values)
low_bps = np.array(low_bps)
high_bps = np.array(high_bps)
mean_bps = np.array(mean_bps)

low_coef = np.corrcoef(ptt_values, low_bps)[0, 1]
low_coef_rec = np.corrcoef(ptt_values_rec, low_bps)[0, 1]
high_coef = np.corrcoef(ptt_values, high_bps)[0, 1]
high_coef_rec = np.corrcoef(ptt_values_rec, high_bps)[0, 1]
mean_coef = np.corrcoef(ptt_values, mean_bps)[0, 1]
mean_coef_rec = np.corrcoef(ptt_values_rec, mean_bps)[0, 1]

print("Correlation Coefficients:")
print(f"PTT - DBP: {low_coef:.2f}")
print(f"PTT - SBP: {high_coef:.2f}")
print(f"PTT - DBP-SBP Mean: {mean_coef:.2f}")
print(f"1/PTT - DBP: {low_coef_rec:.2f}")
print(f"1/PTT - SBP: {high_coef_rec:.2f}")
print(f"1/PTT - DBP-SBP Mean: {mean_coef_rec:.2f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
low_bp_ax, high_bp_ax, mean_bp_ax = axes[0]
low_bp_rec_ax, high_bp_rec_ax, mean_bp_rec_ax = axes[1]

ptt_maxmin = [min(ptt_values), max(ptt_values)]

low_bp_ax.scatter(ptt_values, low_bps, label='DBP', color='blue', alpha=0.7)
low_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, low_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
low_bp_rec_ax.scatter(ptt_values_rec, low_bps, label='DBP', color='blue', alpha=0.7)
high_bp_ax.scatter(ptt_values, high_bps, label='SBP', color='red', alpha=0.7)
high_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, high_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
high_bp_rec_ax.scatter(ptt_values_rec, high_bps, label='SBP', color='red', alpha=0.7)
mean_bp_ax.scatter(ptt_values, mean_bps, label='DBP-SBP Mean', color='green', alpha=0.7)
mean_bp_ax.plot(ptt_maxmin, np.poly1d(np.polyfit(ptt_values, mean_bps, 1))(ptt_maxmin), color='orange', linestyle='--', label='Fit Line')
mean_bp_rec_ax.scatter(ptt_values_rec, mean_bps, label='DBP-SBP Mean', color='green', alpha=0.7)

high_bp_ax.set_title(f'PTT - SBP')
high_bp_ax.set_xlabel('PTT (seconds)')
high_bp_ax.set_ylabel('Blood Pressure (mmHg)')
high_bp_ax.legend()
high_bp_ax.grid()

low_bp_ax.set_title(f'PTT - DBP')
low_bp_ax.set_xlabel('PTT (seconds)')
low_bp_ax.set_ylabel('Blood Pressure (mmHg)')
low_bp_ax.legend()
low_bp_ax.grid()

mean_bp_ax.set_title(f'PTT - DBP-SBP Mean')
mean_bp_ax.set_xlabel('PTT (seconds)')
mean_bp_ax.set_ylabel('Blood Pressure (mmHg)')
mean_bp_ax.legend()
mean_bp_ax.grid()

high_bp_rec_ax.set_title(f'1/PTT - SBP')
high_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
high_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
high_bp_rec_ax.legend()
high_bp_rec_ax.grid()

low_bp_rec_ax.set_title(f'1/PTT - DBP')
low_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
low_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
low_bp_rec_ax.legend()
low_bp_rec_ax.grid()

mean_bp_rec_ax.set_title(f'1/PTT - DBP-SBP Mean')
mean_bp_rec_ax.set_xlabel('1/PTT (1/seconds)')
mean_bp_rec_ax.set_ylabel('Blood Pressure (mmHg)')
mean_bp_rec_ax.legend()
mean_bp_rec_ax.grid()

plt.show()

# write ptt and bp to csv

with open('ptt_bp.csv', 'w', newline='') as csvfile:
    fieldnames = ['ptt', 'low_blood_pressure', 'high_blood_pressure']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ptt, low_bp, high_bp in ptt_bp:
        if ptt is not None:
            writer.writerow({'ptt': ptt, 'low_blood_pressure': low_bp, 'high_blood_pressure': high_bp})


