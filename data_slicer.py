import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path


class RPPGInterpolator:
    def __init__(self, poly_order=3):
        self.poly_order = poly_order
    
    def interpolate(self, time, rppg):
        """
        对跳变的rPPG信号进行插值，生成平滑信号
        """
        if len(time) < 2:
            return rppg
        
        unique_indices = self._find_change_points(rppg)
        
        if len(unique_indices) < 2:
            return rppg
        
        time_unique = time[unique_indices]
        rppg_unique = rppg[unique_indices]
        
        try:
            if len(unique_indices) <= self.poly_order:
                interpolator = interp1d(time_unique, rppg_unique, 
                                       kind='linear', 
                                       fill_value='extrapolate')
            else:
                interpolator = interp1d(time_unique, rppg_unique, 
                                       kind=self.poly_order, 
                                       fill_value='extrapolate')
            
            rppg_smooth = interpolator(time)
            return rppg_smooth
            
        except Exception as e:
            print(f"Warning: Interpolation failed, using linear: {e}")
            interpolator = interp1d(time_unique, rppg_unique, 
                                   kind='linear', 
                                   fill_value='extrapolate')
            return interpolator(time)
    
    def _find_change_points(self, signal, tolerance=1e-9):
        """
        找到信号的跳变点（台阶状信号的角点）
        """
        change_points = [0]
        
        for i in range(1, len(signal)):
            if abs(signal[i] - signal[i-1]) > tolerance:
                change_points.append(i)
        
        if change_points[-1] != len(signal) - 1:
            change_points.append(len(signal) - 1)
        
        return change_points


class SignalResampler:
    def __init__(self, target_fs=512):
        """
        信号重采样器，将信号重采样到指定采样率
        
        参数:
            target_fs: 目标采样频率 (Hz)，默认512Hz
        """
        self.target_fs = target_fs
    
    def resample(self, time, *signals):
        """
        将信号重采样到等间隔的目标采样率
        
        参数:
            time: 原始时间戳数组
            *signals: 一个或多个信号数组
            
        返回:
            time_resampled: 重采样后的时间戳
            *signals_resampled: 重采样后的信号
        """
        if len(time) < 2:
            return (time,) + signals
        
        # 创建等间隔的新时间戳
        start_time = time[0]
        end_time = time[-1]
        duration = end_time - start_time
        
        num_samples = int(duration * self.target_fs) + 1
        time_resampled = np.linspace(start_time, end_time, num_samples)
        
        # 对每个信号进行插值重采样
        signals_resampled = []
        for signal in signals:
            if len(signal) != len(time):
                raise ValueError(f"Signal length ({len(signal)}) must match time length ({len(time)})")
            
            # 使用三次样条插值进行重采样
            interpolator = interp1d(time, signal, kind='cubic', 
                                   fill_value='extrapolate', bounds_error=False)
            signal_resampled = interpolator(time_resampled)
            signals_resampled.append(signal_resampled)
        
        return (time_resampled,) + tuple(signals_resampled)


class ButterworthFilter:
    def __init__(self, lowcut=0.5, highcut=3.0, order=4, fs=None):
        """
        Butterworth带通滤波器
        
        参数:
            lowcut: 低频截止频率 (Hz)，默认0.5Hz对应30bpm
            highcut: 高频截止频率 (Hz)，默认4.0Hz对应240bpm
            order: 滤波器阶数
            fs: 采样频率 (Hz)，如果为None则自动从数据中计算
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.fs = fs
    
    def filter(self, signal, time):
        """
        对信号进行Butterworth带通滤波
        """
        if len(signal) < self.order * 3:
            print(f"Warning: Signal too short for filtering (length: {len(signal)})")
            return signal
        
        # 计算采样频率
        if self.fs is None:
            if len(time) < 2:
                return signal
            dt = np.diff(time)
            fs = 1.0 / np.mean(dt)
        else:
            fs = self.fs
        
        # 检查截止频率是否合理
        nyquist = fs / 2.0
        if self.highcut >= nyquist:
            print(f"Warning: highcut ({self.highcut}Hz) >= Nyquist ({nyquist:.2f}Hz), adjusting...")
            highcut = nyquist * 0.95
        else:
            highcut = self.highcut
        
        if self.lowcut >= highcut:
            print(f"Warning: Invalid filter range, skipping filtering")
            return signal
        
        try:
            # 设计Butterworth带通滤波器
            b, a = butter(self.order, [self.lowcut, highcut], 
                         btype='band', fs=fs)
            
            # 使用filtfilt进行零相位滤波
            filtered_signal = filtfilt(b, a, signal)
            
            return filtered_signal
            
        except Exception as e:
            print(f"Warning: Filtering failed: {e}, returning original signal")
            return signal


class ECGFilter:
    def __init__(self, fs=512):
        """
        ECG专用滤波器，保留QRS波群的高频成分
        
        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        # ECG的R峰含有较高频率成分(5-15Hz)，因此使用更宽的带通范围
        self.lowcut = 0.5   # 去除基线漂移
        self.highcut = 40.0  # 保留R峰的高频成分，去除高频噪声
        self.order = 4
    
    def filter(self, signal):
        """
        对ECG信号进行带通滤波
        """
        if len(signal) < self.order * 3:
            print(f"Warning: ECG signal too short for filtering")
            return signal
        
        nyquist = self.fs / 2.0
        highcut = min(self.highcut, nyquist * 0.95)
        
        try:
            # 设计Butterworth带通滤波器
            b, a = butter(self.order, [self.lowcut, highcut], 
                         btype='band', fs=self.fs)
            
            # 使用filtfilt进行零相位滤波
            filtered_signal = filtfilt(b, a, signal)
            
            return filtered_signal
            
        except Exception as e:
            print(f"Warning: ECG filtering failed: {e}")
            return signal


class PTTEstimator:
    def __init__(self, fs=512):
        """
        脉搏传输时间(PTT)估算器
        
        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        self.ecg_filter = ECGFilter(fs=fs)
    
    def find_peaks(self, signal, signal_type='rppg', min_distance=None):
        """
        在信号中寻找峰值，针对不同信号类型使用不同策略
        
        参数:
            signal: 输入信号
            signal_type: 信号类型 ('rppg' 或 'ecg')
            min_distance: 峰值之间的最小距离（样本点数），默认为0.5秒
        """
        if min_distance is None:
            min_distance = int(self.fs * 0.35)
        
        if signal_type == 'ecg':
            # ECG R峰检测：使用更严格的参数
            # 1. 计算信号的标准差和平均值
            signal_std = np.std(signal)
            signal_mean = np.mean(signal)
            
            # 2. 设置高度阈值（只检测足够高的峰）
            height_threshold = 1.4 * signal_std
            
            # 3. 使用prominence参数确保只检测尖锐的峰（R峰）
            # prominence表示峰的突出度，R峰通常很尖锐
            prominence_threshold = 0.3 * signal_std
            
            # 4. 使用width参数限制峰的宽度（R峰通常较窄）
            # 最大宽度约为0.12秒（QRS波群的典型宽度）
            max_width = int(self.fs * 0.12)
            
            peaks, properties = find_peaks(
                signal,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold,
                width=(1, max_width)
            )
        else:
            # rPPG峰检测：使用较宽松的参数
            peaks, _ = find_peaks(signal, distance=min_distance, height=0)
        
        return peaks
    
    def estimate_ptt(self, time, rppg_signal, ecg_signal):
        """
        估算PTT值
        
        参数:
            time: 时间戳数组
            rppg_signal: rPPG信号
            ecg_signal: ECG信号（原始）
            
        返回:
            ptt: 估算的PTT值（秒），如果无法估算则返回None
            rppg_peaks: rPPG峰值索引
            ecg_peaks: ECG峰值索引（在滤波后信号上）
            matched_pairs: 匹配的峰值对列表 [(ecg_idx, rppg_idx), ...]
            ecg_filtered: 滤波后的ECG信号
        """
        # 对ECG信号进行滤波
        ecg_filtered = self.ecg_filter.filter(ecg_signal)
        
        # 在滤波后的信号上寻找峰值
        rppg_peaks = self.find_peaks(rppg_signal, signal_type='rppg')
        ecg_peaks = self.find_peaks(ecg_filtered, signal_type='ecg')
        
        if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
            return None, rppg_peaks, ecg_peaks, [], ecg_filtered
        
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

                # 只接受合理范围内的PTT (0.02s到0.5s，即20ms到500ms)
                if 0.02 < ptt < 0.5:
                    matched_pairs.append((ecg_idx, rppg_idx))
                    ptt_values.append(ptt)
        
        if len(ptt_values) == 0:
            return None, rppg_peaks, ecg_peaks, [], ecg_filtered
        
        # 使用中位数来避免异常值的影响
        ptt_median = np.median(ptt_values)
        
        # 过滤掉偏差过大的值
        ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < 0.1]
        
        if len(ptt_filtered) == 0:
            return None, rppg_peaks, ecg_peaks, matched_pairs, ecg_filtered
        
        ptt_final = np.mean(ptt_filtered)
        
        return ptt_final, rppg_peaks, ecg_peaks, matched_pairs, ecg_filtered


class DataSegment:
    def __init__(self, time, rppg, ecg, source_file, segment_idx):
        self.time = time
        self.rppg = rppg
        self.ecg = ecg
        self.source_file = source_file
        self.segment_idx = segment_idx
        
    def get_duration(self):
        if len(self.time) < 2:
            return 0
        return self.time[-1] - self.time[0]


class DataSlicer:
    def __init__(self, segment_duration=10.0, target_fs=512):
        self.segment_duration = segment_duration
        self.target_fs = target_fs
        self.interpolator = RPPGInterpolator()
        self.resampler = SignalResampler(target_fs=target_fs)
        self.rppg_filter = ButterworthFilter(lowcut=0.5, highcut=4.0, order=3, fs=target_fs)
        self.ecg_filter = ECGFilter(fs=target_fs)
        
    def load_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df['Time'].values, df['rPPG'].values, df['ECG'].values
        except Exception as e:
            try:
                df = pd.read_csv(file_path)
                return df['Time'].values, df['rppg'].values, df['ecg'].values
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return None, None, None
    
    def slice_data(self, time, rppg, ecg, source_file):
        segments = []
        if len(time) < 2:
            return segments
        
        print(f"  Interpolating rPPG signal...")
        rppg_smooth = self.interpolator.interpolate(time, rppg)
        
        print(f"  Resampling to {self.target_fs}Hz...")
        time_resampled, rppg_resampled, ecg_resampled = self.resampler.resample(
            time, rppg_smooth, ecg
        )
        
        print(f"  Filtering signals...")
        rppg_filtered = self.rppg_filter.filter(rppg_resampled, time_resampled)
        ecg_filtered = self.ecg_filter.filter(ecg_resampled)
        
        start_idx = 0
        segment_idx = 0
        
        while start_idx < len(time_resampled):
            start_time = time_resampled[start_idx]
            end_idx = start_idx
            
            while end_idx < len(time_resampled) and (time_resampled[end_idx] - start_time) < self.segment_duration:
                end_idx += 1
            
            if end_idx >= len(time_resampled):
                end_idx = len(time_resampled)
            
            duration = time_resampled[end_idx - 1] - start_time
            
            if duration >= self.segment_duration - 0.1:
                segment = DataSegment(
                    time=time_resampled[start_idx:end_idx],
                    rppg=rppg_filtered[start_idx:end_idx],
                    ecg=ecg_filtered[start_idx:end_idx],
                    source_file=source_file,
                    segment_idx=segment_idx
                )
                segments.append(segment)
                segment_idx += 1
            
            start_idx = end_idx
        
        return segments
    
    def process_file(self, file_path):
        """处理单个文件，返回其切片"""
        print(f"Processing {file_path.name}...")
        time, rppg, ecg = self.load_csv(file_path)
        
        if time is None:
            return []
        
        segments = self.slice_data(time, rppg, ecg, file_path.name)
        print(f"  Found {len(segments)} segments")
        
        return segments


class SegmentVisualizer:
    def __init__(self, fs=512):
        self.current_segment = None
        self.user_decision = None
        self.fig = None
        self.axes = {}
        self.buttons = {}
        self.ptt_estimator = PTTEstimator(fs=fs)
        
    def _init_plot(self):
        plt.ion()
        self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        self.axes['rppg'] = ax1
        self.axes['ecg'] = ax2
        
        plt.subplots_adjust(bottom=0.15)
        
        ax_accept = plt.axes([0.7, 0.05, 0.1, 0.04])
        ax_reject = plt.axes([0.82, 0.05, 0.1, 0.04])
        
        self.buttons['accept'] = plt.Button(ax_accept, 'Accept')
        self.buttons['reject'] = plt.Button(ax_reject, 'Reject')
        
        self.buttons['accept'].on_clicked(self._on_accept)
        self.buttons['reject'].on_clicked(self._on_reject)
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def _on_accept(self, event):
        self.user_decision = 'accept'
    
    def _on_reject(self, event):
        self.user_decision = 'reject'
    
    def _plot_segment(self, segment):
        self.axes['rppg'].clear()
        self.axes['ecg'].clear()
        
        relative_time = segment.time - segment.time[0]
        
        # 估算PTT并找到峰值（注意：segment.ecg已经是滤波后的信号）
        ptt, rppg_peaks, ecg_peaks, matched_pairs, ecg_filtered = self.ptt_estimator.estimate_ptt(
            relative_time, segment.rppg, segment.ecg
        )
        
        # 绘制rPPG信号和峰值
        self.axes['rppg'].plot(relative_time, segment.rppg, 'b-', linewidth=0.8, label='rPPG')
        if len(rppg_peaks) > 0:
            self.axes['rppg'].plot(
                relative_time[rppg_peaks], 
                segment.rppg[rppg_peaks], 
                'rx', markersize=8, linewidth=2, label=f'rPPG Peaks ({len(rppg_peaks)})'
            )
        
        self.axes['rppg'].set_ylabel('rPPG', fontsize=12)
        
        # 标题包含PTT信息
        ptt_str = f'PTT: {ptt*1000:.1f}ms from {len(matched_pairs)} pairs' if ptt is not None and len(matched_pairs) > 8 else 'PTT: N/A'
        self.axes['rppg'].set_title(
            f'Source: {segment.source_file} | Segment: {segment.segment_idx} | '
            f'Duration: {segment.get_duration():.2f}s | Points: {len(segment.time)} | {ptt_str}',
            fontsize=10
        )
        self.axes['rppg'].grid(True, alpha=0.3)
        self.axes['rppg'].legend(loc='upper right', fontsize=9)
        
        # 绘制ECG信号和峰值（使用滤波后的信号）
        self.axes['ecg'].plot(relative_time, ecg_filtered, 'r-', linewidth=0.8, label='ECG (filtered)')
        if len(ecg_peaks) > 0:
            self.axes['ecg'].plot(
                relative_time[ecg_peaks], 
                ecg_filtered[ecg_peaks], 
                'go', markersize=6, linewidth=2, label=f'R Peaks ({len(ecg_peaks)})'
            )
        
        # 标注匹配的峰值对
        if len(matched_pairs) > 0:
            for ecg_idx, rppg_idx in matched_pairs:
                # 在ECG图上绘制从ECG峰到rPPG峰的连接线
                self.axes['ecg'].plot(
                    [relative_time[ecg_idx], relative_time[rppg_idx]],
                    [ecg_filtered[ecg_idx], ecg_filtered[ecg_idx]],
                    'c--', alpha=0.3, linewidth=1
                )
        
        self.axes['ecg'].set_ylabel('ECG', fontsize=12)
        self.axes['ecg'].set_xlabel('Time (s)', fontsize=12)
        self.axes['ecg'].grid(True, alpha=0.3)
        self.axes['ecg'].legend(loc='upper right', fontsize=9)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def show_segment(self, segment):
        if self.fig is None:
            self._init_plot()
        
        self.current_segment = segment
        self.user_decision = None
        
        self._plot_segment(segment)
        
        while self.user_decision is None:
            plt.pause(0.1)
            if not plt.fignum_exists(self.fig.number):
                self.user_decision = 'reject'
                break
        
        return self.user_decision

    def no_plot_show_segment(self, segment):
        relative_time = segment.time - segment.time[0]
        ptt, rppg_peaks, ecg_peaks, matched_pairs, ecg_filtered = self.ptt_estimator.estimate_ptt(
                relative_time, segment.rppg, segment.ecg
            )
        self.user_decision = 'accept' if ptt is not None and len(matched_pairs) > 5 else 'reject'

        return self.user_decision
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)


class SegmentSaver:
    def __init__(self, output_folder):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.file_counters = {}
    
    def _get_output_filename(self, source_file):
        """
        从源文件名提取patient ID，生成输出文件名
        例如: patient_000015.csv 或 patient_000015_1.csv -> patient_000015_x.csv
        """
        base_name = Path(source_file).stem
        
        # 提取patient ID（去除可能已存在的后缀序号）
        parts = base_name.split('_')
        if len(parts) >= 2:
            # 检查最后一部分是否是数字（已有的序号）
            if parts[-1].isdigit():
                # 如果是，去掉它，只保留patient_xxxxxx部分
                patient_id = '_'.join(parts[:-1])
            else:
                # 如果不是，整个就是patient_xxxxxx
                patient_id = base_name
        else:
            patient_id = base_name
        
        # 为这个patient ID初始化计数器
        if patient_id not in self.file_counters:
            existing_files = list(self.output_folder.glob(f'{patient_id}_*.csv'))
            if existing_files:
                max_idx = max([
                    int(f.stem.split('_')[-1]) 
                    for f in existing_files 
                    if f.stem.split('_')[-1].isdigit()
                ] + [0])
                self.file_counters[patient_id] = max_idx + 1
            else:
                self.file_counters[patient_id] = 1
        
        idx = self.file_counters[patient_id]
        self.file_counters[patient_id] += 1
        
        return f'{patient_id}_{idx}.csv'
    
    def save_segment(self, segment):
        output_filename = self._get_output_filename(segment.source_file)
        output_path = self.output_folder / output_filename
        
        df = pd.DataFrame({
            'Time': segment.time,
            'rPPG': segment.rppg,
            'ECG': segment.ecg
        })
        
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_filename}")


class DataSlicerPipeline:
    def __init__(self, input_folder, output_folder, segment_duration=10.0, target_fs=512, 
                 starting_point=0, ending_point=None):
        self.input_folder = Path(input_folder)
        self.output_folder = output_folder
        self.segment_duration = segment_duration
        self.target_fs = target_fs
        self.starting_point = starting_point
        self.ending_point = ending_point
        
        self.slicer = DataSlicer(segment_duration, target_fs=target_fs)
        self.visualizer = SegmentVisualizer(fs=target_fs)
        self.saver = SegmentSaver(output_folder)
    
    def _extract_patient_num(self, filename):
        """从文件名中提取patient编号"""
        try:
            # 假设文件名格式为 patient_XXXXXX.csv 或 patient_XXXXXX_Y.csv
            parts = filename.stem.split('_')
            if len(parts) >= 2:
                # 尝试提取第二部分作为patient编号
                patient_num = int(parts[1])
                return patient_num
        except (ValueError, IndexError):
            pass
        return None
    
    def run(self):
        print(f"Processing CSV files from: {self.input_folder}")
        print(f"Segment duration: {self.segment_duration}s")
        print(f"Target sampling rate: {self.target_fs}Hz")
        print(f"Patient range: {self.starting_point} to {self.ending_point if self.ending_point else 'end'}")
        print(f"Output folder: {self.output_folder}\n")
        
        csv_files = sorted(self.input_folder.glob('*.csv'))
        
        if not csv_files:
            print("No CSV files found!")
            return
        
        # 过滤文件
        filtered_files = []
        for csv_file in csv_files:
            patient_num = self._extract_patient_num(csv_file)
            if patient_num is None:
                continue
            
            if patient_num < self.starting_point:
                continue
            if self.ending_point is not None and patient_num > self.ending_point:
                continue
            
            filtered_files.append(csv_file)
        
        if not filtered_files:
            print("No CSV files match the patient range!")
            return
        
        print(f"Found {len(filtered_files)} CSV files matching patient range\n")
        
        total_accepted = 0
        total_rejected = 0
        
        for file_idx, csv_file in enumerate(filtered_files, 1):
            print(f"\n{'='*60}")
            print(f"File {file_idx}/{len(filtered_files)}: {csv_file.name}")
            print(f"{'='*60}")
            
            segments = self.slicer.process_file(csv_file)
            
            if not segments:
                print("No valid segments in this file, skipping...")
                continue
            
            print(f"\nReviewing {len(segments)} segments from {csv_file.name}...\n")
            
            file_accepted = 0
            file_rejected = 0
            
            for seg_idx, segment in enumerate(segments, 1):
                print(f"Segment {seg_idx}/{len(segments)} (File {file_idx}/{len(filtered_files)})...")
                # decision = self.visualizer.show_segment(segment)
                decision = self.visualizer.no_plot_show_segment(segment)
                
                if decision == 'accept':
                    self.saver.save_segment(segment)
                    file_accepted += 1
                    total_accepted += 1
                else:
                    file_rejected += 1
                    total_rejected += 1
                    print(f"Rejected: {segment.source_file} segment {segment.segment_idx}")
            
            print(f"\nFile {csv_file.name} complete: {file_accepted} accepted, {file_rejected} rejected")
        
        self.visualizer.close()
        
        print(f"\n{'='*60}")
        print(f"All files processed!")
        print(f"Total accepted: {total_accepted}")
        print(f"Total rejected: {total_rejected}")
        print(f"Total segments: {total_accepted + total_rejected}")
        print(f"{'='*60}")


def main():
    input_folder = input("Enter input folder path: ").strip()
    output_folder = input("Enter output folder pat" \
    "h: ").strip()
    
    try:
        segment_duration = float(input("Enter segment duration in seconds (default 10.0): ").strip() or "10.0")
    except ValueError:
        segment_duration = 10.0
        print("Invalid input, using default 10.0 seconds")

    try:
        target_fs = int(input("Enter target sampling rate in Hz (default 512): ").strip() or "512")
    except ValueError:
        target_fs = 512
        print("Invalid input, using default 512 Hz")
    
    starting_point_input = input("Enter starting patient number (default 0): ").strip()
    starting_point = int(starting_point_input) if starting_point_input.isdigit() else 0
    
    ending_point_input = input("Enter ending patient number (default None): ").strip()
    ending_point = int(ending_point_input) if ending_point_input.isdigit() else None
    
    pipeline = DataSlicerPipeline(input_folder, output_folder, segment_duration, target_fs,
                                   starting_point, ending_point)
    pipeline.run()


if __name__ == '__main__':
    main()
