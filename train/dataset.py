import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import find_peaks

class RPPG2ECGDataset(Dataset):
    def __init__(self, X_rppg_list, Y_ecg_list, window_samples, fs=512, auto_shift=False):
        self.windows = []
        for r, e in zip(X_rppg_list, Y_ecg_list):
            n = len(r)
            if auto_shift:
                # 自动对齐RPPG和ECG信号
                r_peaks, _ = find_peaks(r, distance=0.4*fs, height=np.mean(r))
                e_peaks, _ = find_peaks(e, distance=0.4*fs, height=np.mean(e)+0.5*np.std(e))
                if len(r_peaks) > 0 and len(e_peaks) > 0:
                    shift = 0
                    for i in range(2, min(len(r_peaks), len(e_peaks)) - 1):
                        shift += (r_peaks[i] - e_peaks[i])
                    shift = int(shift / (min(len(r_peaks), len(e_peaks) - 3)))
                    print(f"Auto shift applied: {shift} samples")
                    if shift > 0:
                        r = r[shift:]
                        e = e[:len(e)-shift]
                    elif shift < 0:
                        e = e[-shift:]
                        r = r[:len(r)+shift]
                    n = min(len(r), len(e))
                    r = r[:n]
                    e = e[:n]
            step = window_samples // 2
            for i in range(0, n-window_samples+1, step):
                xr = np.array(r[i:i+window_samples])
                ye = np.array(e[i:i+window_samples])
                self.windows.append((xr.astype('float32'), ye.astype('float32')))

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        xr, ye = self.windows[idx]
        return torch.from_numpy(xr).unsqueeze(0), torch.from_numpy(ye).unsqueeze(0)  # (1,L)
