import os
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import glob
import argparse
import sys
import neurokit2 as nk
from ecg.ecg_process import ECGProcess

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def notch_filter(data, fs=512, freq=50.0, quality=30.0):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, quality)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

def find_rppg_peaks(signal_data, fs=512, min_distance=None):
    """Find peaks in rPPG signal."""
    if min_distance is None:
        min_distance = int(fs * 0.35)
    peaks, _ = signal.find_peaks(signal_data, distance=max(min_distance, 1), height=0)
    return peaks

def calculate_ptt(time, rppg_signal, ecg_signal, ecg_peaks, fs=512):
    """Calculate Pulse Transit Time (PTT) from ECG and rPPG signals."""
    rppg_peaks = find_rppg_peaks(rppg_signal, fs=fs)
    
    if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
        return None, None
    
    matched_pairs = []
    ptt_values = []
    
    for ecg_idx in ecg_peaks:
        if ecg_idx >= len(time):
            continue
        ecg_time = time[ecg_idx]
        future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]
        
        if len(future_rppg_peaks) > 0:
            rppg_idx = future_rppg_peaks[0]
            if rppg_idx >= len(time):
                continue
            rppg_time = time[rppg_idx]
            ptt = rppg_time - ecg_time

            if 0.05 < ptt < 0.4:
                matched_pairs.append((ecg_idx, rppg_idx))
                ptt_values.append(ptt)
    
    if len(ptt_values) == 0:
        return None, None
    
    ptt_median = np.median(ptt_values)
    ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
    
    if len(ptt_filtered) == 0:
        return None, None
    
    ptt_final = np.mean(ptt_filtered)
    std = np.std(ptt_filtered)
    
    return ptt_final, std

class AutoWasher:
    def __init__(self, data_dir, output_dir, reference_dir, patient_info_csv=None, threshold = {'rPPG': 0.75, 'ECG': 0.6}, visualize=False, ecg_method='reference'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.patient_info_csv = patient_info_csv
        self.threshold = threshold
        self.visualize = visualize
        self.ecg_method = ecg_method
        self.segment_length_threshold = 6
        self.fs = 512
        self.mixture_ratio = 0.7
        
        if self.visualize:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            plt.subplots_adjust(bottom=0.25, right=0.85)
            
            self.ax_ecg_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])
            self.ax_rppg_thresh = plt.axes([0.25, 0.05, 0.5, 0.03])
            
            self.slider_ecg = None
            self.slider_rppg = None
            self.ax1_twin = None
            
            self.last_key = None
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Initialize ECG Processor
        if ECGProcess:
            self.ecg_processor = ECGProcess(method='pt', fs=self.fs)
        else:
            self.ecg_processor = None

        self.ecg_refs = []
        self.rppg_refs = []
        self._load_references()
        
        # Storage for PTT results
        self.ptt_results = []  # List of dicts: {patient_id, ptt, std, num_blocks}
        
        # Storage for cleaned patient info
        self.cleaned_patient_info = []  # List of dicts with all patient metrics
        
        # Load patient info lookup
        self.patient_info_lookup = {}
        if patient_info_csv and os.path.exists(patient_info_csv):
            try:
                patient_df = pd.read_csv(patient_info_csv, dtype=int)
                for _, row in patient_df.iterrows():
                    lab_id = int(row.get('lab_patient_id', 0))
                    self.patient_info_lookup[lab_id] = {
                        'hospital_patient_id': row.get('hospital_patient_id', ''),
                        'low_blood_pressure': row.get('low_blood_pressure', -1),
                        'high_blood_pressure': row.get('high_blood_pressure', -1)
                    }
                print(f"Loaded patient info for {len(self.patient_info_lookup)} patients.")
            except Exception as e:
                print(f"Error loading patient info CSV: {e}")

    def _on_key(self, event):
        self.last_key = event.key

    def _load_references(self):
        ecg_dir = os.path.join(self.reference_dir, 'ecg')
        rppg_dir = os.path.join(self.reference_dir, 'rppg')
        
        self.ecg_refs = self._load_ref_from_dir(ecg_dir, 'ecg')
        self.rppg_refs = self._load_ref_from_dir(rppg_dir, 'rppg')
        
        if not self.ecg_refs:
            self.ecg_refs.extend(self._load_ref_from_dir(self.reference_dir, 'ecg'))
        if not self.rppg_refs:
            self.rppg_refs.extend(self._load_ref_from_dir(self.reference_dir, 'rppg'))
        
        if not self.ecg_refs and os.path.exists('./reference_ecg'):
            print("  [Info] Falling back to ./reference_ecg for ECG references")
            self.ecg_refs.extend(self._load_ref_from_dir('./reference_ecg', 'ecg'))
        
        print(f"Loaded {len(self.ecg_refs)} ECG references and {len(self.rppg_refs)} rPPG references.")

    def _load_ref_from_dir(self, directory, signal_col_name):
        refs = []
        if not os.path.exists(directory):
            return refs
            
        for f in os.listdir(directory):
            if f.endswith(".csv"):
                try:
                    path = os.path.join(directory, f)
                    df = pd.read_csv(path)
                    cols = [c for c in df.columns if signal_col_name.lower() in c.lower()]
                    if cols:
                        sig = df[cols[0]].to_numpy()
                        refs.append(sig)
                except Exception as e:
                    print(f"Error loading reference {f}: {e}")
        return refs

    def process_all(self):
        """Main processing loop."""
        patient_dirs = glob.glob(os.path.join(self.data_dir, "patient_*"))
        for p_dir in patient_dirs:
            self.process_patient(p_dir)
        
        # Save PTT results to CSV
        if self.ptt_results:
            ptt_csv = os.path.join(self.output_dir, 'ptt_results.csv')
            ptt_df = pd.DataFrame(self.ptt_results)
            ptt_df.to_csv(ptt_csv, index=False)
            print(f"\n[PTT Results] Saved to {ptt_csv}")
            print(f"  Total patients with PTT: {len(ptt_df[ptt_df['ptt'].notna()])}")
            if len(ptt_df[ptt_df['ptt'].notna()]) > 0:
                print(f"  Mean PTT: {ptt_df['ptt'].mean():.4f}s")
                print(f"  PTT Range: {ptt_df['ptt'].min():.4f}s - {ptt_df['ptt'].max():.4f}s")
        
        # Save cleaned patient info to CSV
        if self.cleaned_patient_info:
            cleaned_csv = os.path.join(self.output_dir, 'cleaned_patient_info.csv')
            cleaned_df = pd.DataFrame(self.cleaned_patient_info)
            cleaned_df.to_csv(cleaned_csv, index=False, float_format='%.4f')
            print(f"\n[Cleaned Patient Info] Saved to {cleaned_csv}")
            print(f"  Total patients: {len(cleaned_df)}")

    def process_patient(self, patient_dir):
        """Process a single patient directory."""
        merged_csv = os.path.join(patient_dir, "merged_log.csv")
        if not os.path.exists(merged_csv):
            return

        print(f"Processing {patient_dir}...")
        try:
            df = pd.read_csv(merged_csv)
            # Ensure columns exist
            required = ['Timestamp', 'RPPG', 'ECG']
            if not all(col in df.columns for col in required):
                print(f"Skipping {patient_dir}: Missing columns.")
                return
            
            timestamps = df['Timestamp'].to_numpy()
            ecg_signal = df['ECG'].to_numpy()
            ecg_signal = filter_signal(ecg_signal, fs=self.fs, lowcut=0.5, highcut=30, order=4)
            ecg_signal = notch_filter(ecg_signal, fs=self.fs, freq=50.0, quality=30.0)
            rppg_signal = df['RPPG'].to_numpy()
            rppg_signal = filter_signal(rppg_signal, fs=self.fs, lowcut=0.5, highcut=5.0, order=4)
            
            # Pre-calculate NeuroKit quality vector
            rppg_quality_vec = None
            try:
                # Calculate on full signal to handle windowing correctly
                q = nk.ppg_quality(rppg_signal, sampling_rate=self.fs, method="templatematch")
                if isinstance(q, pd.DataFrame):
                    rppg_quality_vec = q.iloc[:, 0].to_numpy()
                elif isinstance(q, np.ndarray):
                    rppg_quality_vec = q
                else:
                    rppg_quality_vec = np.full(len(rppg_signal), float(q))
            except Exception:
                # Fallback or ignore if NK fails
                pass
            
            # 1. Detect Peaks
            peaks = self._detect_peaks(ecg_signal)
            print(f"  [Debug] Found {len(peaks)} peaks in {os.path.basename(patient_dir)}")
            if len(peaks) < 2:
                print(f"Not enough peaks for {patient_dir}")
                return

            # Pre-calculate ECG quality vector based on selected method
            if self.ecg_method == 'neurokit':
                ecg_quality_vec = self._calculate_ecg_quality_vector_neurokit(ecg_signal)
            elif self.ecg_method == 'mixture':
                ecg_quality_vec_ref = self._calculate_ecg_quality_vector_custom(ecg_signal, peaks)
                ecg_quality_vec_nk = self._calculate_ecg_quality_vector_neurokit(ecg_signal)
                if ecg_quality_vec_ref is not None and ecg_quality_vec_nk is not None:
                    ecg_quality_vec = np.maximum(ecg_quality_vec_ref, ecg_quality_vec_nk)
                    # ecg_quality_vec = self.mixture_ratio * ecg_quality_vec_ref + (1 - self.mixture_ratio) * ecg_quality_vec_nk
                elif ecg_quality_vec_ref is not None:
                    ecg_quality_vec = ecg_quality_vec_ref
                elif ecg_quality_vec_nk is not None:
                    ecg_quality_vec = ecg_quality_vec_nk
                else:
                    ecg_quality_vec = None
            else:
                ecg_quality_vec = self._calculate_ecg_quality_vector_custom(ecg_signal, peaks)

            # 2. Segment and Calculate Similarity
            segments_info = [] # List of dicts: {start_idx, end_idx, is_good}
            sim_ecg_list = []
            sim_rppg_list = []
            
            for i in range(len(peaks)):
                # Define segment window (3:7 ratio)
                peak = peaks[i]
                prev_peak = peaks[i-1] if i > 0 else None
                next_peak = peaks[i+1] if i < len(peaks) - 1 else None
                
                if prev_peak is None or next_peak is None:
                    continue
                
                prev_interval = max(1, peak - prev_peak)
                next_interval = max(1, next_peak - peak)
                
                start = int(peak - 0.3 * prev_interval)
                end = int(peak + 0.7 * next_interval)
                
                start = max(0, start)
                end = min(len(ecg_signal), end)
                
                if end - start <= 1:
                    continue
                
                # Extract segments
                seg_ecg = ecg_signal[start:end]
                seg_time = timestamps[start:end]
                
                # Check similarity (Reference)
                sim_ecg_ref = self._max_similarity(seg_time, seg_ecg, self.ecg_refs)
                
                # Check similarity (Quality Vector-based: NeuroKit or Mixture)
                sim_ecg_vec = 0.0
                if ecg_quality_vec is not None:
                    s_idx = max(0, start)
                    e_idx = min(len(ecg_quality_vec), end)
                    if e_idx > s_idx:
                        sim_ecg_vec = np.min(ecg_quality_vec[s_idx:e_idx])
                
                # Select ECG score based on method
                if self.ecg_method == 'neurokit' or self.ecg_method == 'mixture':
                    sim_ecg = sim_ecg_vec
                else:
                    sim_ecg = sim_ecg_ref
                
                # Calculate rPPG SQI using NeuroKit vector
                sim_rppg = 0.0
                if rppg_quality_vec is not None:
                    s_idx = max(0, start)
                    e_idx = min(len(rppg_quality_vec), end)
                    if e_idx > s_idx:
                        # Use min to ensure the whole segment meets the threshold
                        sim_rppg = np.min(rppg_quality_vec[s_idx:e_idx])
                
                sim_ecg_list.append(sim_ecg)
                sim_rppg_list.append(sim_rppg)
                
                is_good = (sim_ecg >= self.threshold['ECG']) and (sim_rppg >= self.threshold['rPPG'])
                
                segments_info.append({
                    'start': start,
                    'end': end,
                    'is_good': is_good,
                    'sim_ecg': sim_ecg,
                    'sim_ecg_ref': sim_ecg_ref,
                    'sim_ecg_vec': sim_ecg_vec,
                    'sim_rppg': sim_rppg
                })

            if sim_ecg_list:
                print(f"  [Debug] Avg Sim ECG: {np.mean(sim_ecg_list):.3f}, Max: {np.max(sim_ecg_list):.3f}")
                print(f"  [Debug] Avg Sim RPPG: {np.mean(sim_rppg_list):.3f}, Max: {np.max(sim_rppg_list):.3f}")
                good_count = sum(1 for s in segments_info if s['is_good'])
                print(f"  [Debug] Good segments: {good_count}/{len(segments_info)}")

            # 3. Find Continuous Blocks
            blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
            
            if not blocks:
                print(f"No valid blocks found for {patient_dir}")

            # 3.5. Calculate PTT from good blocks
            ptt_mean, ptt_std, ptt_length = self._calculate_ptt_from_blocks(timestamps, ecg_signal, rppg_signal, peaks, blocks)
            
            # 3.6. Calculate HR and HRV from good blocks
            hr_mean, hr_std, sdnn, rmssd = self._calculate_hr_hrv_from_blocks(timestamps, peaks, blocks)
            
            # 4. Visualize (Optional)
            save_blocks = False
            final_ecg_th = self.threshold['ECG']
            final_rppg_th = self.threshold['rPPG']
            
            if self.visualize:
                print(f"  [Debug] Starting visualization for {os.path.basename(patient_dir)}...")
                accepted, final_ecg_th, final_rppg_th = self._visualize_review(timestamps, ecg_signal, rppg_signal, blocks, peaks, segments_info, patient_dir, rppg_quality_vec, ecg_quality_vec, ptt_mean, ptt_std)
                if accepted:
                    # Re-calculate blocks with new thresholds
                    for seg in segments_info:
                        # Update sim_ecg based on method (method might have changed? No, method is fixed in args)
                        # But we can allow changing method in visualization? 
                        # The user removed radio buttons for rPPG. I won't add them for ECG unless asked.
                        # So just use the current method's score.
                        seg['is_good'] = (seg['sim_ecg'] >= final_ecg_th) and (seg['sim_rppg'] >= final_rppg_th)
                    blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
                    # Recalculate PTT with updated blocks
                    ptt_mean, ptt_std, ptt_length = self._calculate_ptt_from_blocks(timestamps, ecg_signal, rppg_signal, peaks, blocks)
                    # Recalculate HR and HRV with updated blocks
                    hr_mean, hr_std, sdnn, rmssd = self._calculate_hr_hrv_from_blocks(timestamps, peaks, blocks)
                    save_blocks = True
            elif blocks:
                save_blocks = True
            
            # 5. Save PTT results
            if blocks:
                patient_id = os.path.basename(patient_dir)
                self.ptt_results.append({
                    'patient_id': patient_id,
                    'ptt': ptt_mean if ptt_mean is not None else np.nan,
                    'std': ptt_std if ptt_std is not None else np.nan,
                    'num_blocks': len(blocks)
                })
                if ptt_mean is not None:
                    print(f"  [PTT] Patient {patient_id}: {ptt_mean:.4f}s ± {ptt_std:.4f}s ({len(blocks)} blocks)")
            
            # 6. Save blocks and patient info
            if save_blocks and blocks:
                self._save_blocks(df, blocks, patient_dir)
                # Save cleaned patient info
                self._save_patient_info(patient_dir, segments_info, blocks, ptt_mean, ptt_std, ptt_length, hr_mean, hr_std, sdnn, rmssd)

        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            import traceback
            traceback.print_exc()

    def _detect_peaks(self, ecg_signal):
        if self.ecg_processor:
            self.ecg_processor.process(ecg_signal)
            peaks = self.ecg_processor.get_peaks()
            return np.array(peaks) if peaks is not None else np.array([])
        else:
            # Fallback simple peak detection
            peaks, _ = signal.find_peaks(ecg_signal, distance=int(self.fs*0.5))
            return peaks
    def _calculate_ecg_quality_vector_neurokit(self, ecg_signal):
        """Calculates ECG quality vector using NeuroKit2."""
        try:
            # Use NeuroKit's zhao2018 method for ECG quality
            q = nk.ecg_quality(ecg_signal, sampling_rate=self.fs, method="templatematch")
            if isinstance(q, pd.DataFrame):
                quality_vec = q.iloc[:, 0].to_numpy()
            elif isinstance(q, np.ndarray):
                quality_vec = q
            else:
                quality_vec = np.full(len(ecg_signal), float(q))
            return quality_vec
        except Exception as e:
            print(f"Error calculating ECG SQI with NeuroKit: {e}")
            return None
    def _calculate_ecg_quality_vector_custom(self, ecg_signal, peaks):
        """Calculates ECG quality vector based on correlation with average beat."""
        try:
            # 1. Extract beats
            # Window: -0.2s to +0.4s (typical QRS-T)
            pre = int(0.2 * self.fs)
            post = int(0.4 * self.fs)
            
            beats = []
            valid_peaks = []
            
            for peak in peaks:
                if peak - pre >= 0 and peak + post < len(ecg_signal):
                    beat = ecg_signal[peak - pre : peak + post]
                    beats.append(beat)
                    valid_peaks.append(peak)
            
            if not beats:
                return None
                
            beats_array = np.array(beats)
            
            # 2. Compute Reference Beat (Median is robust to artifacts)
            ref_beat = np.median(beats_array, axis=0)
            
            # Normalize ref beat
            if np.std(ref_beat) > 1e-6:
                ref_beat = (ref_beat - np.mean(ref_beat)) / np.std(ref_beat)
            else:
                return np.zeros(len(ecg_signal))

            # 3. Correlate each beat
            scores = []
            for i, peak in enumerate(valid_peaks):
                beat = beats_array[i]
                if np.std(beat) > 1e-6:
                    beat_norm = (beat - np.mean(beat)) / np.std(beat)
                    corr = np.corrcoef(beat_norm, ref_beat)[0, 1]
                else:
                    corr = 0.0
                scores.append(max(0.0, corr))
            
            # 4. Interpolate to create vector (Nearest Neighbor)
            indices = np.arange(len(ecg_signal))
            if len(valid_peaks) > 1:
                f = interp1d(valid_peaks, scores, kind='nearest', bounds_error=False, fill_value=(scores[0], scores[-1]))
                quality_vec = f(indices)
            else:
                quality_vec = np.full(len(ecg_signal), scores[0] if scores else 0.0)
            
            return quality_vec

        except Exception as e:
            print(f"Error calculating ECG SQI: {e}")
            return None

    def _max_similarity(self, timestamps, segment_signal, refs):
        if not refs:
            return 0.0
        
        # Resample to fixed length (512)
        target_len = 512
        if len(segment_signal) < 2:
            return 0.0
            
        new_t = np.linspace(timestamps[0], timestamps[-1], target_len)
        resampled = interp1d(timestamps, segment_signal, kind='cubic', fill_value="extrapolate")(new_t)
        
        # Normalize
        resampled = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-6)
        
        max_sim = -1.0
        for ref in refs:
            if len(ref) != target_len:
                # Assuming refs are already 512 length or need resampling?
                # template_matching.py assumes refs are loaded and checked for length.
                # I'll assume refs are 512. If not, I should resample them too, but usually templates are fixed.
                continue
            
            # Normalize ref
            ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-6)
            
            # Correlation
            corr = np.corrcoef(resampled, ref_norm)[0, 1]
            if corr > max_sim:
                max_sim = corr
                
        return max_sim

    def _calculate_ptt_from_blocks(self, timestamps, ecg_signal, rppg_signal, peaks, blocks):
        """Calculate weighted average PTT from all good blocks."""
        if not blocks:
            return None, None, 0
        
        all_ptt_values = []  # Collect all individual PTT datapoints
        block_segment_counts = []  # Track number of segments per block
        
        for block in blocks:
            start_idx = block[0]['start']
            end_idx = block[-1]['end']
            
            # Extract segment data
            seg_timestamps = timestamps[start_idx:end_idx]
            seg_ecg = ecg_signal[start_idx:end_idx]
            seg_rppg = rppg_signal[start_idx:end_idx]
            
            # Find peaks within this segment
            seg_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)] - start_idx
            
            # Get individual PTT values for this block
            ptt_values = self._calculate_ptt_values(seg_timestamps, seg_rppg, seg_ecg, seg_peaks)
            
            if ptt_values:
                all_ptt_values.extend(ptt_values)
                block_segment_counts.append(len(block))
        
        if not all_ptt_values:
            return None, None, 0
        
        # Calculate mean and std from all individual PTT datapoints
        ptt_mean = np.mean(all_ptt_values)
        ptt_std = np.std(all_ptt_values)
        ptt_length = len(all_ptt_values)
        
        return ptt_mean, ptt_std, ptt_length
    
    def _calculate_ptt_values(self, time, rppg_signal, ecg_signal, ecg_peaks):
        """Get individual PTT values (not averaged) from a segment."""
        rppg_peaks = find_rppg_peaks(rppg_signal, fs=self.fs)
        
        if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
            return []
        
        ptt_values = []
        
        for ecg_idx in ecg_peaks:
            if ecg_idx >= len(time):
                continue
            ecg_time = time[ecg_idx]
            future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]
            
            if len(future_rppg_peaks) > 0:
                rppg_idx = future_rppg_peaks[0]
                if rppg_idx >= len(time):
                    continue
                rppg_time = time[rppg_idx]
                ptt = rppg_time - ecg_time

                if 0.05 < ptt < 0.4:
                    ptt_values.append(ptt)
        
        if not ptt_values:
            return []
        
        # Filter outliers based on median
        ptt_median = np.median(ptt_values)
        ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
        
        return ptt_filtered

    def _calculate_hr_hrv_from_blocks(self, timestamps, peaks, blocks):
        """Calculate HR and HRV metrics from all good blocks."""
        if not blocks:
            return None, None, None, None
        
        all_rr_intervals = []
        
        for block in blocks:
            start_idx = block[0]['start']
            end_idx = block[-1]['end']
            
            # Find peaks within this block
            block_peaks = peaks[(peaks >= start_idx) & (peaks < end_idx)]
            
            if len(block_peaks) < 2:
                continue
            
            # Calculate RR intervals in seconds
            peak_times = timestamps[block_peaks]
            rr_intervals = np.diff(peak_times)
            
            # Filter physiologically plausible RR intervals (30-200 BPM -> 0.3-2.0s)
            valid_rr = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
            all_rr_intervals.extend(valid_rr)
        
        if len(all_rr_intervals) < 2:
            return None, None, None, None
        
        rr_array = np.array(all_rr_intervals)
        
        # Calculate HR (beats per minute)
        hr_mean = 60.0 / np.mean(rr_array)
        hr_std = 60.0 * np.std(rr_array) / (np.mean(rr_array) ** 2)
        
        # Calculate HRV metrics
        # SDNN: Standard deviation of NN intervals (in ms)
        sdnn = np.std(rr_array) * 1000
        
        # RMSSD: Root mean square of successive differences (in ms)
        successive_diff = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(successive_diff ** 2)) * 1000
        
        return hr_mean, hr_std, sdnn, rmssd
    
    def _find_continuous_blocks(self, segments_info, min_len=None):
        if min_len is None:
            min_len = self.segment_length_threshold
        blocks = []
        current_block = []
        
        for i, seg in enumerate(segments_info):
            if seg['is_good']:
                if not current_block:
                    current_block.append(seg)
                else:  
                    last_seg = current_block[-1]
                    if abs(seg['start'] - last_seg['end']) < 2:
                        current_block.append(seg)
                    else:
                        if len(current_block) >= min_len:
                            blocks.append(current_block)
                        current_block = [seg]
            else:
                if len(current_block) >= min_len:
                    blocks.append(current_block)
                current_block = []
        
        if len(current_block) >= min_len:
            blocks.append(current_block)
            
        return blocks

    def _visualize_review(self, timestamps, ecg, rppg, blocks, peaks, segments_info, patient_dir, rppg_quality_vec=None, ecg_quality_vec=None, ptt=None, ptt_std=None):
        ax1, ax2 = self.axes
        ax1.clear()
        ax2.clear()
        
        # Normalize RPPG for visualization
        rppg_vis = rppg.copy()
        if len(rppg_vis) > 0:
            r_min, r_max = np.min(rppg_vis), np.max(rppg_vis)
            if r_max > r_min:
                rppg_vis = (rppg_vis - r_min) / (r_max - r_min)
        
        # Plot signals
        ax1.plot(timestamps, ecg, label='ECG', color='black', alpha=0.5)
        ax2.plot(timestamps, rppg_vis, label='RPPG (Norm)', color='black', alpha=0.5)
        
        # Plot NeuroKit Quality Vector if available
        if rppg_quality_vec is not None:
            ax2.plot(timestamps[:len(rppg_quality_vec)], rppg_quality_vec, color='purple', alpha=0.8, label='NK Quality', linestyle='-')
            
        # Plot ECG Quality Vector if available
        if ecg_quality_vec is not None:
            if self.ax1_twin is None:
                self.ax1_twin = ax1.twinx()
            else:
                self.ax1_twin.clear()
            self.ax1_twin.plot(timestamps[:len(ecg_quality_vec)], ecg_quality_vec, color='purple', alpha=0.8, label='ECG Quality', linestyle='-')
            self.ax1_twin.set_ylim(0, 1.1)
            self.ax1_twin.set_ylabel('ECG Quality', color='purple')
        elif self.ax1_twin is not None:
            self.ax1_twin.clear()
            self.ax1_twin.set_visible(False)
        
        ax2.set_ylim(-0.1, 1.1)
        
        # Mark ECG peaks
        if len(peaks) > 0:
            valid_peaks = peaks[peaks < len(timestamps)]
            ax1.plot(timestamps[valid_peaks], ecg[valid_peaks], 'r.', markersize=5, label='Peaks')
        
        # Mark rPPG peaks
        rppg_peaks = find_rppg_peaks(rppg, fs=self.fs)
        if len(rppg_peaks) > 0:
            valid_rppg_peaks = rppg_peaks[rppg_peaks < len(timestamps)]
            ax2.plot(timestamps[valid_rppg_peaks], rppg_vis[valid_rppg_peaks], 'r.', markersize=5, label='Peaks')

        # Mark segment boundaries (static)
        for seg in segments_info:
            t_start = timestamps[seg['start']]
            ax1.axvline(x=t_start, color='blue', alpha=0.05, linestyle='-', linewidth=0.5)

        # Add PTT to title if available
        patient_name = os.path.basename(patient_dir)
        if ptt is not None and ptt_std is not None:
            ax1.set_title(f"Patient: {patient_name} - ECG | PTT: {ptt:.4f}s ± {ptt_std:.4f}s")
        else:
            ax1.set_title(f"Patient: {patient_name} - ECG | PTT: N/A")
        ax2.set_title("RPPG")
        ax1.legend(loc='upper right')
        plt.suptitle("Adjust thresholds. Press 'y' to accept, 'n' to reject.")

        # Sliders
        self.ax_ecg_thresh.clear()
        self.ax_rppg_thresh.clear()
        
        self.slider_ecg = Slider(self.ax_ecg_thresh, 'ECG Thresh', 0.0, 1.0, valinit=self.threshold['ECG'])
        self.slider_rppg = Slider(self.ax_rppg_thresh, 'RPPG Thresh', 0.0, 1.0, valinit=self.threshold['rPPG'])

        ax2.legend(loc='upper right')
        self.current_spans = []

        def update(val):
            # Remove old spans
            for span in self.current_spans:
                span.remove()
            self.current_spans = []
            
            ecg_th = self.slider_ecg.val
            rppg_th = self.slider_rppg.val
            
            # Update segments status
            for seg in segments_info:
                rppg_score = seg['sim_rppg']
                
                seg['is_good'] = (seg['sim_ecg'] >= ecg_th) and (rppg_score >= rppg_th)
                
                t_start = timestamps[seg['start']]
                t_end = timestamps[seg['end']]
                
                if seg['sim_ecg'] >= ecg_th:
                    span = ax1.axvspan(t_start, t_end, color='yellow', alpha=0.2)
                    self.current_spans.append(span)
                if rppg_score >= rppg_th:
                    span = ax2.axvspan(t_start, t_end, color='yellow', alpha=0.2)
                    self.current_spans.append(span)

            # Find blocks
            current_blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
            
            if current_blocks:
                for i, block in enumerate(current_blocks):
                    start_idx = block[0]['start']
                    end_idx = block[-1]['end']
                    t_start = timestamps[start_idx]
                    t_end = timestamps[end_idx]
                    
                    span1 = ax1.axvspan(t_start, t_end, color='green', alpha=0.4)
                    span2 = ax2.axvspan(t_start, t_end, color='green', alpha=0.4)
                    self.current_spans.extend([span1, span2])
            
            self.fig.canvas.draw_idle()

        self.slider_ecg.on_changed(update)
        self.slider_rppg.on_changed(update)
        
        # Initial draw
        update(None)
        
        self.last_key = None
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Wait for key
        try:
            while True:
                plt.pause(0.1)
                if self.last_key == 'y':
                    return True, self.slider_ecg.val, self.slider_rppg.val
                elif self.last_key == 'n':
                    return False, self.threshold['ECG'], self.threshold['rPPG']
        except Exception:
            return False, self.threshold['ECG'], self.threshold['rPPG']

    def _save_patient_info(self, patient_dir, segments_info, blocks, ptt, ptt_std, ptt_length, hr_mean=None, hr_std=None, sdnn=None, rmssd=None):
        """Save cleaned patient information to cleaned_patient_info list."""
        patient_id = os.path.basename(patient_dir)  # patient_xxxxxx
        
        # Extract lab patient ID (e.g., "000002" from "patient_000002")
        lab_patient_id = int(patient_id.replace('patient_', ''))
        
        # Calculate average SQI for accepted segments
        good_segments = [seg for seg in segments_info if seg['is_good']]
        
        if not good_segments:
            return
        
        ecg_sqi_avg = np.mean([seg['sim_ecg'] for seg in good_segments])
        rppg_sqi_avg = np.mean([seg['sim_rppg'] for seg in good_segments])
        
        # Look up patient info
        patient_info = self.patient_info_lookup.get(lab_patient_id, {})
        hospital_patient_id = patient_info.get('hospital_patient_id', '')
        low_bp = patient_info.get('low_blood_pressure', -1)
        high_bp = patient_info.get('high_blood_pressure', -1)
        if hospital_patient_id == '':
            print(f"  [Warning] No hospital_patient_id found for lab_patient_id {lab_patient_id}")
        # Append to cleaned patient info
        self.cleaned_patient_info.append({
            'Lab_Patient_ID': lab_patient_id,
            'Hospital_Patient_ID': hospital_patient_id,
            'ECG_SQI_AVG': ecg_sqi_avg,
            'rPPG_SQI_AVG': rppg_sqi_avg,
            'PTT': ptt if ptt is not None else np.nan,
            'PTT_STDDEV': ptt_std if ptt_std is not None else np.nan,
            'PTT_LENGTH': ptt_length,
            'HR_MEAN': hr_mean if hr_mean is not None else np.nan,
            'HR_STD': hr_std if hr_std is not None else np.nan,
            'HRV_SDNN': sdnn if sdnn is not None else np.nan,
            'HRV_RMSSD': rmssd if rmssd is not None else np.nan,
            'Low_Blood_Pressure': low_bp,
            'High_Blood_Pressure': high_bp
        })
    
    def _save_blocks(self, df, blocks, patient_dir):
        patient_id = os.path.basename(patient_dir) # patient_xxxxxx
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        for i, block in enumerate(blocks):
            start_idx = block[0]['start']
            end_idx = block[-1]['end']
            
            # Slice DataFrame
            block_df = df.iloc[start_idx:end_idx].copy()
            
            # Z-score normalization
            # Change: Don't normalize RPPG, ECG normalize by dividing 32768
            for col in ['RPPG', 'ECG']:
                if col in block_df.columns:
                    if col == 'ECG':
                        block_df[col] = block_df[col] / 32768.0  # Normalize ECG by dividing by 32768
                    elif col == 'RPPG':
                        if block_df[col].std() > 1e-6:
                            pass # Don't normalize RPPG, just keep it as is. The model can learn from the raw values.
                            # block_df[col] = (block_df[col] - block_df[col].mean()) / block_df[col].std()
                        else:
                            block_df[col] = 0.0
            
            filename = f"{patient_id}_{i+1}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            block_df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")

if __name__ == "__main__":
    mirror_id = 1
    parser = argparse.ArgumentParser(description="Auto Wash Patient Data")
    parser.add_argument("--data_dir", type=str, default=f"./mirror{mirror_id}_data", help="Directory containing patient folders")
    parser.add_argument("--output_dir", type=str, default=f"./mirror{mirror_id}_auto_cleaned", help="Directory to save cleaned segments")
    parser.add_argument("--reference_dir", type=str, default="./reference_signals", help="Directory containing reference signals")
    parser.add_argument("--patient_info_csv", type=str, default=f"./merged_patient_info_{mirror_id}.csv", help="CSV file with patient info including blood pressure")
    parser.add_argument("--threshold_ecg", type=float, default=0.6, help="ECG similarity threshold")
    parser.add_argument("--threshold_rppg", type=float, default=0.75, help="rPPG similarity threshold")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization and manual review")
    parser.add_argument("--ecg_method", type=str, default="reference", choices=["reference", "neurokit", "mixture"], help="Method for ECG quality assessment")
    
    args = parser.parse_args()
    
    washer = AutoWasher(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        patient_info_csv=args.patient_info_csv,
        threshold={'ECG': args.threshold_ecg, 'rPPG': args.threshold_rppg},
        visualize=False,
        ecg_method='mixture'
    )
    
    washer.process_all()
