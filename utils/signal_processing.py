import numpy as np
import scipy.signal as signal


def filter_signal(data, fs=512, lowcut=0.5, highcut=5.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def notch_filter(data, fs=512, freq=50.0, quality=30.0):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, quality)
    return signal.filtfilt(b, a, data)


def find_rppg_peaks(signal_data, fs=512, min_distance=None):
    if min_distance is None:
        min_distance = int(fs * 0.35)
    peaks, _ = signal.find_peaks(signal_data, distance=max(min_distance, 1), height=0)
    return peaks


def calculate_ptt(timestamps, rppg_signal, ecg_signal, ecg_peaks, fs=512,
                  ptt_lo=0.05, ptt_hi=0.4, filter_margin=0.1):
    """Calculate pulse transit time (PTT) from matched ECG→rPPG peak pairs.

    Returns (ptt_mean, ptt_std) or (None, None) if insufficient data.
    """
    rppg_peaks = find_rppg_peaks(rppg_signal, fs=fs)

    if len(rppg_peaks) == 0 or len(ecg_peaks) == 0:
        return None, None

    ptt_values = []
    for ecg_idx in ecg_peaks:
        if ecg_idx >= len(timestamps):
            continue
        ecg_time = timestamps[ecg_idx]
        future_rppg_peaks = rppg_peaks[rppg_peaks > ecg_idx]

        if len(future_rppg_peaks) > 0:
            rppg_idx = future_rppg_peaks[0]
            if rppg_idx >= len(timestamps):
                continue
            rppg_time = timestamps[rppg_idx]
            ptt = rppg_time - ecg_time

            if ptt_lo < ptt < ptt_hi:
                ptt_values.append(ptt)

    if len(ptt_values) == 0:
        return None, None

    ptt_median = np.median(ptt_values)
    ptt_filtered = [p for p in ptt_values if abs(p - ptt_median) < filter_margin]

    if len(ptt_filtered) == 0:
        return None, None

    return float(np.mean(ptt_filtered)), float(np.std(ptt_filtered))
