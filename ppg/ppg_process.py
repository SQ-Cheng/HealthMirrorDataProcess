import numpy as np
import scipy.signal as signal
import pandas as pd
import neurokit2 as nk

def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    data = signal.filtfilt(b, a, data)
    return data

class PPGProcess:
    def __init__(self, method='nk', fs=512, log=False):
        self.method = method
        self.fs = fs
        self.result_df = None
        self.peaks = None
        self.additional_signals = None
        
    def process(self, ppg_signal):
        if self.method == 'nk':
            return self._process_neurokit(ppg_signal)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _process_neurokit(self, ppg_signal):
        nk_signals, info = nk.ppg_process(ppg_signal, sampling_rate=self.fs)
        self.result_df = nk_signals
        self.peaks = nk_signals['PPG_Peaks'].values.nonzero()[0]
        self.additional_signals = {'quality': nk_signals['PPG_Quality'].values}
        return self.result_df