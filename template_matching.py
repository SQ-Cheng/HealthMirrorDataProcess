from data.patient_info import PatientInfo
from data.load_data import DataLoader
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import csv
import os
import pandas as pd
import neurokit2 as nk
from ecg.ecg_process import ECGProcess

lab = False

if lab:
    data_dir = "./lab_mirror_data"
    output_file = "lab_overall_patient_info.csv"
    merged_patient_file = "lab_merged_patient_info.csv"
    cleaned_dir = "./lab_test_cleaned"
    reference_dir = "./lab_reference_signals"
else:
    data_dir = "./mirror1_data"
    output_file = "overall_patient_info.csv"
    merged_patient_file = "merged_patient_info.csv"
    cleaned_dir = "./test_sliced"
    reference_dir = "./reference_signals"

def load_all_patients(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    return patient_info_list

def load_patient_with_bp(data_dir=data_dir, output_file=output_file):
    patient_info = PatientInfo(data_dir, save_dir=output_file, mode="file")
    patient_info_list = patient_info.extract(data_file=merged_patient_file)
    patient_with_bp = [p for p in patient_info_list if int(p['low_blood_pressure']) != -1 and int(p['high_blood_pressure']) != -1]
    return patient_with_bp

def load_data_for_patients(patient_list, raw_dir=data_dir, cleaned_dir=cleaned_dir):
    patient_ids = [int(p['lab_patient_id']) for p in patient_list]
    data_loader = DataLoader(raw_dir=raw_dir, cleaned_dir=cleaned_dir)
    raw_data_loader = data_loader.load_raw_data(patient_id=patient_ids)
    cleaned_data_loader = data_loader.load_cleaned_data(patient_id=patient_ids)
    return raw_data_loader, cleaned_data_loader

def load_reference_waveforms(ref_dir):
    for f in os.listdir(ref_dir):
        if f.endswith(".csv"):
            try:
                file_path = os.path.join(ref_dir, f)
                df = pd.read_csv(file_path)
                timestamps = df.loc[:, 'timestamps'].astype(float).to_numpy()
                ecg_signal = df.loc[:, 'ecg'].astype(float).to_numpy()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            yield int(f[4:7]), timestamps, ecg_signal

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

def find_rppg_peaks(signal_data, fs=512, min_distance=None):
    if min_distance is None:
        min_distance = int(fs * 0.35)
    peaks, properties = signal.find_peaks(signal_data, distance=min_distance, height=0)
    return peaks

def calculate_ptt(time, rppg_signal, ecg_signal, ecg_processor):
    rppg_peaks = find_rppg_peaks(rppg_signal, fs=512)
    ecg_processor.process(ecg_signal)
    ecg_peaks = ecg_processor.get_peaks()
    
    if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
        return None, None, None
    
    matched_pairs = []
    ptt_values = []
    
    for ecg_idx in ecg_peaks:
        ecg_time = time[ecg_idx]
        future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]
        
        if len(future_rppg_peaks) > 0:
            rppg_idx = future_rppg_peaks[0]
            rppg_time = time[rppg_idx]
            ptt = rppg_time - ecg_time

            if 0.05 < ptt < 0.4:
                matched_pairs.append((ecg_idx, rppg_idx))
                ptt_values.append(ptt)
    
    if len(ptt_values) == 0:
        return None, None, None
    
    ptt_median = np.median(ptt_values)
    ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
    
    if len(ptt_filtered) == 0:
        return None, None, None
    
    ptt_final = np.mean(ptt_filtered)
    std = np.std(ptt_filtered)
    
    return ptt_final, None, std

class RawSignalViewer:
    def __init__(self, dataloader, reference_waveforms=None, method='nk', reference_dir='./reference_signals'):
        self.dataloader = dataloader
        self.reference_waveforms = reference_waveforms or []
        print (f"Loaded {len(self.reference_waveforms)} reference waveforms.")
        self.reference_dir = reference_dir
        os.makedirs(self.reference_dir, exist_ok=True)
        self.dataframe = None
        self.current_raw_idx = 0
        self.current_patient_id = None
        self.current_ptt = None
        self.current_std = None
        self.fig, self.axes = plt.subplots(3, 2, figsize=(18, 12))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ptt_results = []
        self.fs = 512
        self.segment_window = 0.7
        self.ecg_processor = ECGProcess(method=method, fs=self.fs)
        self.ecg_peaks = np.array([])

        self.selected_signal = 0
        self.signal_list = ['ecg', 'rppg', 'ppg']
        self.signal_sources = {
            'ecg': {'columns': ['ECG', 'ecg'], 'label': 'ECG'},
            'rppg': {'columns': ['RPPG', 'rppg', 'rPPG'], 'label': 'rPPG'},
            'ppg': {'columns': ['PPG_IR', 'ppg_ir', 'IR', 'ppg_ir_norm'], 'label': 'PPG IR'},
        }
        self.clipped_segments = {s: [] for s in self.signal_list}
        self.current_segment_idx = {s: 0 for s in self.signal_list}
        self._highlight_selected_row()
        
    def on_key_press(self, event):
        if event.key == 'y':
            print("Accept")
            if self.current_ptt is not None:
                self.ptt_results.append((self.current_patient_id, self.current_ptt, None, self.current_std))
            self._advance_record()
        elif event.key == 'n':
            print("Reject")
            self._advance_record()
        elif event.key == 'esc':
            print("Quit")
            plt.close(self.fig)
        elif event.key == 'up':
            self._reset_row_facecolor(self.selected_signal)
            self.selected_signal = (self.selected_signal - 1) % len(self.signal_list)
            self._highlight_selected_row()
            print(f"Selected {self.signal_list[self.selected_signal].upper()} signal")
        elif event.key == 'down':
            self._reset_row_facecolor(self.selected_signal)
            self.selected_signal = (self.selected_signal + 1) % len(self.signal_list)
            self._highlight_selected_row()
            print(f"Selected {self.signal_list[self.selected_signal].upper()} signal")
        elif event.key == 'right':
            print("Next Segment")
            self.current_segment_idx[self.signal_list[self.selected_signal]] += 1
            self.update_subplot(signal_idx=self.selected_signal, segment=True)
        elif event.key == 'w':
            print("Save segment as reference")
            self.save_reference_segment(signal_idx=self.selected_signal)  
    
    def update_subplot(self, signal_idx=0, segment=False):
        signal_name = self.signal_list[signal_idx]
        if self.dataframe is None:
            return
        label = self.signal_sources[signal_name]['label']
        timestamps = self._get_timestamps()
        values = self._get_signal_values(signal_name)
        if values.size == 0 or timestamps.size == 0:
            self.axes[signal_idx, 0].clear()
            self.axes[signal_idx, 0].set_title(f'{label} signal unavailable')
            self.axes[signal_idx, 1].clear()
            self.axes[signal_idx, 1].set_title('No segments available')
            self.fig.canvas.draw_idle()
            return

        if not segment:
            left_ax = self.axes[signal_idx, 0]
            left_ax.clear()
            peaks = self._detect_peaks(signal_name, values)
            left_ax.plot(timestamps, values, label=f'{label} Signal')
            if peaks.size:
                left_ax.plot(timestamps[peaks], values[peaks], 'x', label='Peaks')
            left_ax.set_title(f'{label} Signal - Patient {self.current_patient_id}')
            left_ax.set_xlabel('Time (s)')
            left_ax.set_ylabel('Amplitude')
            left_ax.legend()
            left_ax.grid(True, alpha=0.3)
            if signal_name == 'rppg' and self.ecg_peaks.size:
                valid = self.ecg_peaks[self.ecg_peaks < values.size]
                if valid.size:
                    left_ax.plot(timestamps[valid], values[valid], '|', label='ECG Peaks', color='orange', markersize=8)
                    left_ax.legend()

            if signal_name == 'ecg':
                self._annotate_ecg_plot(left_ax, timestamps, values)
            self._store_segments(signal_name, timestamps, values, peaks)

        self._plot_segment(signal_name, signal_idx, label)
        
    def save_reference_segment(self, signal_idx=0):
        signal_name = self.signal_list[signal_idx]
        segments = self.clipped_segments.get(signal_name, [])
        if not segments:
            print(f"No segments available for {signal_name}")
            return
        idx = self.current_segment_idx[signal_name] % len(segments)
        seg_df = segments[idx]
        
        target_length = 512
        new_timestamps = np.linspace(seg_df['Timestamp'].iloc[0], seg_df['Timestamp'].iloc[-1], target_length)
        resampled = interp1d(seg_df['Timestamp'], seg_df['Value'], kind='cubic', fill_value='extrapolate')(new_timestamps)
        
        signal_dir = os.path.join(self.reference_dir, signal_name)
        os.makedirs(signal_dir, exist_ok=True)
        
        filename = f"ref_{str(self.current_patient_id).zfill(3)}_{signal_name}.csv"
        filepath = os.path.join(signal_dir, filename)
        
        save_df = pd.DataFrame({
            'timestamps': new_timestamps,
            signal_name: resampled
        })
        save_df.to_csv(filepath, index=False)
        print(f"Saved reference {signal_name} segment to {filepath}")

    def update_plot(self):
        for idx in range(len(self.signal_list)):
            self.update_subplot(signal_idx=idx, segment=False)
        self._highlight_selected_row()
        self.fig.suptitle("Press 'y' to accept, 'n' to reject, 'esc' to quit", fontsize=12)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
    
    def __call__(self):
        self.current_patient_id, self.dataframe = next(self.dataloader, (None, None))
        if self.dataframe is None:
            plt.close(self.fig)
            return
        self.current_segment_idx = {s: 0 for s in self.signal_list}
        self.clipped_segments = {s: [] for s in self.signal_list}
        self.ecg_peaks = np.array([])
        self.update_plot()
        plt.show()

        return self.current_raw_idx + 1

    def _advance_record(self):
        self.current_raw_idx += 1
        self.current_segment_idx = {s: 0 for s in self.signal_list}
        self.clipped_segments = {s: [] for s in self.signal_list}
        self.current_patient_id, self.dataframe = next(self.dataloader, (None, None))
        self.current_ptt = None
        self.current_std = None
        self.ecg_peaks = np.array([])
        if self.dataframe is None:
            plt.close(self.fig)
            return
        self.update_plot()

    def _reset_row_facecolor(self, row):
        self.axes[row, 0].set_facecolor('white')
        self.axes[row, 1].set_facecolor('white')

    def _highlight_selected_row(self):
        for row in range(len(self.signal_list)):
            color = 'lightgray' if row == self.selected_signal else 'white'
            self.axes[row, 0].set_facecolor(color)
            self.axes[row, 1].set_facecolor(color)
        self.fig.canvas.draw_idle()

    def _get_timestamps(self):
        for key in ['Timestamp', 'Time', 'timestamp', 'timestamps']:
            if key in self.dataframe:
                return self.dataframe[key].to_numpy()
        return np.array([])

    def _get_signal_values(self, signal_name):
        for column in self.signal_sources[signal_name]['columns']:
            if column in self.dataframe:
                return self.dataframe[column].to_numpy()
        return np.array([])

    def _detect_peaks(self, signal_name, values):
        if signal_name == 'ecg':
            self.ecg_processor.process(values)
            peaks_list = self.ecg_processor.get_peaks()
            if peaks_list is None:
                self.ecg_peaks = np.array([])
                return np.array([])
            peaks = np.asarray(peaks_list, dtype=int)
            self.ecg_peaks = peaks if peaks.size else np.array([])
            return self.ecg_peaks
        distance = int(self.fs * 0.35)
        peaks, _ = signal.find_peaks(values, distance=max(distance, 1), height=0)
        return peaks

    def _store_segments(self, signal_name, timestamps, values, peaks):
        if signal_name == 'ecg':
            segments = self._clip_ecg_segments(timestamps, values, peaks)
        else:
            segments = self._clip_default_segments(timestamps, values, peaks)
        self.clipped_segments[signal_name] = segments
        self.current_segment_idx[signal_name] = 0

    def _clip_ecg_segments(self, timestamps, values, peaks):
        if peaks.size < 2:
            return self._clip_default_segments(timestamps, values, peaks)
        segments = []
        for idx, peak in enumerate(peaks):
            prev_peak = peaks[idx - 1] if idx > 0 else None
            next_peak = peaks[idx + 1] if idx < peaks.size - 1 else None
            if prev_peak is None or next_peak is None:
                continue
            prev_interval = max(1, peak - prev_peak)
            next_interval = max(1, next_peak - peak)
            start = int(peak - 0.3 * prev_interval)
            end = int(peak + 0.7 * next_interval)
            start = max(0, start)
            end = min(values.size, end)
            if end - start <= 1:
                continue
            segments.append(pd.DataFrame({'Timestamp': timestamps[start:end], 'Value': values[start:end]}))
        if not segments:
            return self._clip_default_segments(timestamps, values, peaks)
        return segments

    def _clip_default_segments(self, timestamps, values, peaks):
        half_window = max(1, int(self.segment_window * self.fs) // 2)
        segments = []
        target_peaks = peaks if peaks.size else np.array([np.argmax(values)])
        for peak in target_peaks:
            start = max(0, peak - half_window)
            end = min(values.size, peak + half_window)
            if end - start <= 1:
                continue
            segments.append(pd.DataFrame({'Timestamp': timestamps[start:end], 'Value': values[start:end]}))
        return segments

    def _plot_segment(self, signal_name, signal_idx, label):
        right_ax = self.axes[signal_idx, 1]
        right_ax.clear()
        segments = self.clipped_segments.get(signal_name, [])
        if not segments:
            right_ax.set_title('No segments available')
            right_ax.grid(True, alpha=0.3)
            self.fig.canvas.draw_idle()
            return
        idx = self.current_segment_idx[signal_name] % len(segments)
        seg_df = segments[idx]
        right_ax.plot(seg_df['Timestamp'], seg_df['Value'], label=f'{label} Segment')
        base_title = f'{label} Segment {idx + 1}/{len(segments)} - Patient {self.current_patient_id}'
        if signal_name == 'ecg':
            final_title = self._annotate_ecg_segment(seg_df, base_title)
        else:
            final_title = base_title
        right_ax.set_xlabel('Time (s)')
        right_ax.set_ylabel('Amplitude')
        right_ax.legend()
        right_ax.grid(True, alpha=0.3)
        right_ax.set_title(final_title)
        self.fig.canvas.draw_idle()

    def _annotate_ecg_plot(self, axis, timestamps, values):
        rppg_values = self._get_signal_values('rppg')
        if rppg_values.size == 0:
            self.current_ptt = None
            self.current_std = None
            return
        ptt, _, std = calculate_ptt(timestamps, rppg_values, values, self.ecg_processor)
        self.current_ptt = ptt
        self.current_std = std
        additional = self.ecg_processor.get_additional_signals()
        if self.ecg_processor.method == 'pt' and additional:
            pantompkins = additional.get('pantompkins')
            pt_peaks = additional.get('pt_peaks', [])
            if pantompkins is not None:
                axis.plot(timestamps, pantompkins, label='Pan-Tompkins', alpha=0.6)
                if pt_peaks is not None and len(pt_peaks):
                    axis.plot(timestamps[pt_peaks], pantompkins[pt_peaks], 's', label='PT Peaks', markersize=4)
        if self.ecg_processor.method == 'nk' and additional:
            quality = additional.get('quality')
            if quality is not None:
                axis.plot(timestamps, quality, label='ECG Quality', alpha=0.6)
                mean_quality = np.mean(quality)
            else:
                mean_quality = None
        else:
            mean_quality = None
        if ptt is not None:
            title = f'ECG Signal - PTT {ptt:.3f}s, Std {std:.3f}'
        else:
            title = 'ECG Signal - PTT: N/A'
        if mean_quality is not None:
            title += f', Mean quality: {mean_quality:.3f}'
        axis.set_title(title)

    def _annotate_ecg_segment(self, segment, base_title):
        if not self.reference_waveforms:
            return base_title
        target_length = 512
        new_timestamps = np.linspace(segment['Timestamp'].iloc[0], segment['Timestamp'].iloc[-1], target_length)
        resampled = interp1d(segment['Timestamp'], segment['Value'], kind='cubic', fill_value='extrapolate')(new_timestamps)
        sims = []
        cosines = []
        for patient_id, _, ref_ecg in self.reference_waveforms:
            if len(ref_ecg) != target_length:
                continue
            sims.append(np.corrcoef(resampled, ref_ecg)[0, 1])
            denom = np.linalg.norm(resampled) * np.linalg.norm(ref_ecg)
            if denom == 0:
                continue
            cosines.append(np.dot(resampled, ref_ecg) / denom)
        if not sims:
            return base_title
        cos_mean = np.mean(cosines) if cosines else float('nan')
        return f"{base_title} | Linear {np.mean(sims):.3f}, Cosine {cos_mean:.3f}"


if __name__ == '__main__':
    bp_patient_list = load_patient_with_bp()
    raw_data_loader, cleaned_data_loader = load_data_for_patients(bp_patient_list)
    reference_waveforms = list(load_reference_waveforms('./reference_ecg'))
    
    viewer = RawSignalViewer(cleaned_data_loader, reference_waveforms=reference_waveforms, method='pt', reference_dir=reference_dir)
    total_processed = viewer()
    print(f"Total processed raw signals: {total_processed}")
    
