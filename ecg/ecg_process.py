import scipy.signal as signal
import numpy as np
import pandas as pd
import neurokit2 as nk

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

class ECGProcess:
    def __init__(self, method='nk', fs=512, log=False):
        self.method = method
        self.fs = fs
        self.result_df = None
        self.peaks = None
        self.additional_signals = None
        
    def process(self, ecg_signal):
        if self.method == 'nk':
            return self._process_neurokit(ecg_signal)
        elif self.method == 'pt':
            return self._process_pantompkins(ecg_signal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _process_neurokit(self, ecg_signal):
        nk_signals, info = nk.ecg_process(ecg_signal, sampling_rate=self.fs)
        self.result_df = nk_signals
        self.peaks = nk_signals['ECG_R_Peaks'].values.nonzero()[0]
        self.additional_signals = {'quality': nk_signals['ECG_Quality'].values}
        return self.result_df
    
    def _process_pantompkins(self, ecg_signal):
        filtered_ecg = filter_signal(ecg_signal, fs=self.fs, lowcut=5.0, highcut=15.0, order=2)
        derivative_ecg = np.gradient(filtered_ecg)
        squared_ecg = np.square(derivative_ecg)
        
        window_size = int(0.150 * self.fs)
        pantompkins = np.convolve(squared_ecg, np.ones(window_size)/window_size, mode='same')
        pantompkins = 2 * pantompkins / np.max(pantompkins)
        
        signal_std = np.std(pantompkins)
        pt_peaks, _ = signal.find_peaks(
            pantompkins,
            distance=int(self.fs * 0.35),
            prominence=0.1 * signal_std,
        )

        peaks = []
        for peak in pt_peaks:
            left = max(0, peak - int(0.1 * self.fs))
            right = min(len(ecg_signal), peak + int(0.1 * self.fs))
            p = np.argmax(ecg_signal[left:right]) + left
            peaks.append(p)
        
        self.peaks = np.array(peaks)
        self.additional_signals = {'pantompkins': pantompkins, 'pt_peaks': pt_peaks}
        
        df_data = {
            'ECG_R_Peaks': np.zeros(len(ecg_signal), dtype=int),
            'PanTompkins': pantompkins
            }
        df_data['ECG_R_Peaks'][self.peaks] = 1
        self.result_df = pd.DataFrame(df_data)
        
        return self.result_df
    
    def get_peaks(self):
        return self.peaks
    
    def get_result_dataframe(self):
        return self.result_df
    
    def get_additional_signals(self):
        return self.additional_signals