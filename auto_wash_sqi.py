import argparse
import glob
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d

from ecg.ecg_process import ECGProcess


RAW_SIGNAL_COLUMNS = {
    "ecg": ["timestamp", "ecg"],
    "rppg": ["timestamp", "rppg"],
    "ppg": ["timestamp", "ppg_red", "ppg_ir", "ppg_green"],
}


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


def compute_snr_db(x, fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15):
    """Estimate narrow-band SNR (dB) around the dominant frequency in a band."""
    x = np.asarray(x)
    if len(x) < 4:
        return -100.0

    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2

    hi_hz = min(hi_hz, float(freqs[-1]) - 1e-6)
    if hi_hz <= lo_hz:
        return -100.0

    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band):
        return -100.0

    band_spec = spec[band]
    band_freqs = freqs[band]
    peak_freq = band_freqs[np.argmax(band_spec)]

    signal_band = (freqs >= (peak_freq - peak_width_hz)) & (freqs <= (peak_freq + peak_width_hz))
    signal_power = np.sum(spec[signal_band])
    noise_power = np.sum(spec[band]) - signal_power
    noise_power = max(noise_power, 1e-12)

    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def ecg_sqi_autocorr(ecg, fs):
    """Autocorrelation periodicity SQI in [0, 1]."""
    x = np.asarray(ecg, dtype=np.float64)
    if len(x) < 4:
        return 0.0

    x = x - np.mean(x)
    acf = np.correlate(x, x, mode="full")
    acf = acf[len(x) - 1 :]
    if acf[0] <= 1e-12:
        return 0.0
    acf = acf / acf[0]

    lag_lo = max(1, int(fs * 0.33))
    lag_hi = min(len(acf) - 1, int(fs * 1.50))
    if lag_hi <= lag_lo:
        return 0.0

    peak = float(np.max(acf[lag_lo : lag_hi + 1]))
    return float(np.clip(peak, 0.0, 1.0))


def ecg_sqi_beat_to_beat_corr(ecg, fs, target_len=128):
    """Beat-to-beat morphology correlation SQI in [0, 1]."""
    x = np.asarray(ecg, dtype=np.float64)
    if len(x) < int(2.0 * fs):
        return 0.0

    x = x - np.mean(x)
    peaks, _ = signal.find_peaks(x, distance=max(int(0.35 * fs), 1))
    if len(peaks) < 3:
        return 0.0

    beats = []
    for i in range(1, len(peaks) - 1):
        start = int((peaks[i - 1] + peaks[i]) // 2)
        end = int((peaks[i] + peaks[i + 1]) // 2)
        if end - start < 8:
            continue

        beat = x[start:end]
        beat_std = np.std(beat)
        if beat_std < 1e-8:
            continue

        beat = (beat - np.mean(beat)) / beat_std
        beat_rs = signal.resample(beat, target_len)
        beats.append(beat_rs)

    if len(beats) < 2:
        return 0.0

    corrs = []
    for i in range(len(beats) - 1):
        c = np.corrcoef(beats[i], beats[i + 1])[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    if not corrs:
        return 0.0

    corrs = np.asarray(corrs, dtype=np.float64)
    sqi = (corrs + 1.0) * 0.5
    return float(np.clip(np.median(sqi), 0.0, 1.0))


def rppg_sqi_autocorr(rppg, fs):
    """Autocorrelation periodicity SQI in [0, 1] for rPPG."""
    return ecg_sqi_autocorr(rppg, fs)


def snr_db_to_sqi_linear(snr_db, snr_min_db=-8.0, snr_max_db=12.0):
    """Linear mapping of SNR (dB) to SQI in [0, 1].
    
    Args:
        snr_db: SNR value in dB (can be scalar or array)
        snr_min_db: SNR value (dB) that maps to SQI=0.0
        snr_max_db: SNR value (dB) that maps to SQI=1.0
    
    Returns:
        SQI in [0, 1] via linear interpolation, clipped to bounds.
    """
    x = np.asarray(snr_db, dtype=np.float64)
    sqi = (x - snr_min_db) / (snr_max_db - snr_min_db)
    return np.clip(sqi, 0.0, 1.0)


class AutoWasherSQI:
    def __init__(
        self,
        data_dir,
        output_dir,
        reference_dir="./reference_signals",
        patient_info_csv=None,
        threshold=None,
        visualize=False,
        mirror_version="1",
        rppg_weight_snr=0.2,
        rppg_weight_autocorr=0.8,
        ecg_weight_autocorr=0.4,
        ecg_weight_btb_corr=0.3,
        ecg_weight_template=0.3,
        polarity="neg",
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.reference_dir = reference_dir
        self.patient_info_csv = patient_info_csv
        self.threshold = threshold or {"rPPG": 0.4, "ECG": 0.35}
        self.visualize = visualize
        self.mirror_version = str(mirror_version)

        self.segment_length_threshold = 6
        self.fs = 512
        self.sqi_window_sec = 3.0
        self.sqi_step_sec = 0.25
        self.rppg_weight_snr = float(rppg_weight_snr)
        self.rppg_weight_autocorr = float(rppg_weight_autocorr)
        self.ecg_weight_autocorr = float(ecg_weight_autocorr)
        self.ecg_weight_btb_corr = float(ecg_weight_btb_corr)
        self.ecg_weight_template = float(ecg_weight_template)
        self.polarity = polarity

        if self.mirror_version not in {"1", "2"}:
            raise ValueError("mirror_version must be '1' or '2'")

        self.ecg_processor = ECGProcess(method="pt", fs=self.fs) if ECGProcess else None
        self.ecg_refs = []
        self._load_ecg_references()
        self._normalize_rppg_weights()
        self._normalize_ecg_weights()

        if self.visualize:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            plt.subplots_adjust(bottom=0.25, right=0.85)

            self.ax_ecg_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])
            self.ax_rppg_thresh = plt.axes([0.25, 0.05, 0.5, 0.03])

            self.slider_ecg = None
            self.slider_rppg = None
            self.ax1_sqi = None
            self.last_key = None
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.cleaned_patient_info = []

        self.patient_info_lookup = {}
        if patient_info_csv and os.path.exists(patient_info_csv):
            try:
                patient_df = pd.read_csv(patient_info_csv, dtype=int)
                for _, row in patient_df.iterrows():
                    lab_id = int(row.get("lab_patient_id", 0))
                    self.patient_info_lookup[lab_id] = {
                        "hospital_patient_id": row.get("hospital_patient_id", ""),
                        "low_blood_pressure": row.get("low_blood_pressure", -1),
                        "high_blood_pressure": row.get("high_blood_pressure", -1),
                    }
                print(f"Loaded patient info for {len(self.patient_info_lookup)} patients.")
            except Exception as e:
                print(f"Error loading patient info CSV: {e}")

    def _on_key(self, event):
        self.last_key = event.key

    def _normalize_rppg_weights(self):
        weights = np.array(
            [
                max(0.0, self.rppg_weight_snr),
                max(0.0, self.rppg_weight_autocorr),
            ],
            dtype=np.float64,
        )
        if np.sum(weights) <= 1e-12:
            weights = np.array([1.0, 0.0], dtype=np.float64)
        weights = weights / np.sum(weights)
        self.rppg_weight_snr = float(weights[0])
        self.rppg_weight_autocorr = float(weights[1])

    def _normalize_ecg_weights(self):
        weights = np.array(
            [
                max(0.0, self.ecg_weight_autocorr),
                max(0.0, self.ecg_weight_btb_corr),
                max(0.0, self.ecg_weight_template),
            ],
            dtype=np.float64,
        )
        if np.sum(weights) <= 1e-12:
            weights = np.array([0.4, 0.3, 0.3], dtype=np.float64)
        weights = weights / np.sum(weights)
        self.ecg_weight_autocorr = float(weights[0])
        self.ecg_weight_btb_corr = float(weights[1])
        self.ecg_weight_template = float(weights[2])

    def _load_ref_from_dir(self, directory, signal_col_name):
        refs = []
        if not os.path.exists(directory):
            return refs

        for f in os.listdir(directory):
            if not f.endswith(".csv"):
                continue
            try:
                path = os.path.join(directory, f)
                df = pd.read_csv(path)
                cols = [c for c in df.columns if signal_col_name.lower() in str(c).lower()]
                if not cols:
                    continue
                sig = pd.to_numeric(df[cols[0]], errors="coerce").dropna().to_numpy(dtype=np.float64)
                if len(sig) > 8:
                    refs.append(sig)
            except Exception as e:
                print(f"Error loading reference {f}: {e}")
        return refs

    def _load_ecg_references(self):
        ecg_dir = os.path.join(self.reference_dir, "ecg")
        self.ecg_refs = self._load_ref_from_dir(ecg_dir, "ecg")

        if not self.ecg_refs:
            self.ecg_refs.extend(self._load_ref_from_dir(self.reference_dir, "ecg"))

        if not self.ecg_refs and os.path.exists("./reference_ecg"):
            print("[Info] Falling back to ./reference_ecg for ECG template references")
            self.ecg_refs.extend(self._load_ref_from_dir("./reference_ecg", "ecg"))

        print(f"Loaded {len(self.ecg_refs)} ECG template references.")

    def _max_similarity_to_refs(self, segment_signal, refs, target_len=512):
        if not refs:
            return 0.0

        x = np.asarray(segment_signal, dtype=np.float64)
        if len(x) < 4:
            return 0.0

        x = signal.resample(x, target_len)
        x_std = np.std(x)
        if x_std < 1e-8:
            return 0.0
        x = (x - np.mean(x)) / x_std

        max_sim = 0.0
        for ref in refs:
            r = np.asarray(ref, dtype=np.float64)
            if len(r) < 4:
                continue
            if len(r) != target_len:
                r = signal.resample(r, target_len)

            r_std = np.std(r)
            if r_std < 1e-8:
                continue
            r = (r - np.mean(r)) / r_std

            corr = float(np.corrcoef(x, r)[0, 1])
            if np.isfinite(corr):
                max_sim = max(max_sim, corr)

        return float(np.clip(max_sim, 0.0, 1.0))

    def process_all(self):
        patient_dirs = glob.glob(os.path.join(self.data_dir, "patient_*"))
        for p_dir in patient_dirs:
            self.process_patient(p_dir)

        if self.cleaned_patient_info:
            cleaned_csv = os.path.join(self.output_dir, "cleaned_patient_info.csv")
            cleaned_df = pd.DataFrame(self.cleaned_patient_info)
            cleaned_df.to_csv(cleaned_csv, index=False, float_format="%.4f")
            print(f"\n[Cleaned Patient Info] Saved to {cleaned_csv}")
            print(f"  Total patients: {len(cleaned_df)}")

    def process_patient(self, patient_dir):
        print(f"Processing {patient_dir}...")
        try:
            df = self._load_patient_dataframe(patient_dir)
            if df is None:
                return

            timestamps = df["Timestamp"].to_numpy()
            if self.polarity == "neg" and "ECG" in df.columns:
                ecg_signal = -df["ECG"].to_numpy()
            ecg_signal = filter_signal(ecg_signal, fs=self.fs, lowcut=0.5, highcut=30.0, order=4)
            ecg_signal = notch_filter(ecg_signal, fs=self.fs, freq=50.0, quality=30.0)
            rppg_signal = df["RPPG"].to_numpy()
            rppg_signal = filter_signal(rppg_signal, fs=self.fs, lowcut=0.5, highcut=5.0, order=4)

            peaks = self._detect_peaks(ecg_signal)
            print(f"  [Debug] Found {len(peaks)} peaks in {os.path.basename(patient_dir)}")
            if len(peaks) < 2:
                print(f"Not enough peaks for {patient_dir}")
                return

            ecg_autocorr_vec, ecg_btb_vec, rppg_snr_db_vec, rppg_autocorr_vec = self._build_window_quality_vectors(
                ecg_signal,
                rppg_signal,
            )
            ecg_template_vec = self._build_ecg_template_quality_vector(ecg_signal, peaks)
            ecg_autocorr_norm_vec = np.clip(ecg_autocorr_vec, 0.0, 1.0)
            ecg_btb_norm_vec = np.clip(ecg_btb_vec, 0.0, 1.0)
            ecg_template_norm_vec = np.clip(ecg_template_vec, 0.0, 1.0)
            ecg_fused_vec = np.clip(
                self.ecg_weight_autocorr * ecg_autocorr_norm_vec
                + self.ecg_weight_btb_corr * ecg_btb_norm_vec
                + self.ecg_weight_template * ecg_template_norm_vec,
                0.0,
                1.0,
            )
            rppg_snr_norm_vec = snr_db_to_sqi_linear(rppg_snr_db_vec, snr_min_db=-8.0, snr_max_db=12.0)
            rppg_autocorr_norm_vec = np.clip(rppg_autocorr_vec, 0.0, 1.0)
            rppg_fused_vec = self._combine_rppg_quality_vectors(
                rppg_snr_norm_vec,
                rppg_autocorr_norm_vec,
            )

            segments_info = []
            ecg_sqi_list = []
            ecg_autocorr_list = []
            ecg_btb_list = []
            ecg_template_list = []
            rppg_snr_list = []
            rppg_autocorr_list = []
            rppg_final_list = []

            for i in range(len(peaks)):
                peak = peaks[i]
                prev_peak = peaks[i - 1] if i > 0 else None
                next_peak = peaks[i + 1] if i < len(peaks) - 1 else None

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

                seg_ecg = ecg_signal[start:end]
                seg_rppg = rppg_signal[start:end]

                # Robust per-segment SQI: derive from multi-second window quality vectors.
                sim_ecg_autocorr = self._segment_quality_from_single_vector(ecg_autocorr_norm_vec, start, end)
                sim_ecg_btb = self._segment_quality_from_single_vector(ecg_btb_norm_vec, start, end)
                sim_ecg_template = self._segment_quality_from_single_vector(ecg_template_norm_vec, start, end)
                sim_ecg = self._segment_quality_from_single_vector(ecg_fused_vec, start, end)
                if not np.isfinite(sim_ecg):
                    sim_ecg = (
                        self.ecg_weight_autocorr * ecg_sqi_autocorr(seg_ecg, fs=self.fs)
                        + self.ecg_weight_btb_corr * ecg_sqi_beat_to_beat_corr(seg_ecg, fs=self.fs)
                        + self.ecg_weight_template * self._max_similarity_to_refs(seg_ecg, self.ecg_refs)
                    )

                sim_rppg = self._segment_quality_from_single_vector(rppg_snr_db_vec, start, end)
                if not np.isfinite(sim_rppg):
                    sim_rppg = compute_snr_db(seg_rppg, fs=self.fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
                sim_rppg_snr_norm = self._segment_quality_from_single_vector(rppg_snr_norm_vec, start, end)
                sim_rppg_autocorr = self._segment_quality_from_single_vector(rppg_autocorr_norm_vec, start, end)
                sim_rppg_final = self._segment_quality_from_single_vector(
                    rppg_fused_vec,
                    start,
                    end,
                )

                ecg_sqi_list.append(sim_ecg)
                ecg_autocorr_list.append(sim_ecg_autocorr)
                ecg_btb_list.append(sim_ecg_btb)
                ecg_template_list.append(sim_ecg_template)
                rppg_snr_list.append(sim_rppg)
                rppg_autocorr_list.append(sim_rppg_autocorr)
                rppg_final_list.append(sim_rppg_final)

                is_good = (sim_ecg >= self.threshold["ECG"]) and (sim_rppg_final >= self.threshold["rPPG"])
                segments_info.append(
                    {
                        "start": start,
                        "end": end,
                        "is_good": is_good,
                        "sim_ecg": sim_ecg,
                        "sim_ecg_autocorr": sim_ecg_autocorr,
                        "sim_ecg_btb_corr": sim_ecg_btb,
                        "sim_ecg_template": sim_ecg_template,
                        "sim_rppg": sim_rppg_final,
                        "sim_rppg_snr_db": sim_rppg,
                        "sim_rppg_snr_norm": sim_rppg_snr_norm,
                        "sim_rppg_autocorr": sim_rppg_autocorr,
                    }
                )

            if ecg_sqi_list:
                print(
                    "  [Debug] ECG fused SQI (autocorr + btb + template): "
                    f"mean={np.mean(ecg_sqi_list):.3f}, max={np.max(ecg_sqi_list):.3f}"
                )
                ecg_ac_vals = [v for v in ecg_autocorr_list if np.isfinite(v)]
                if ecg_ac_vals:
                    print(
                        "  [Debug] ECG autocorr SQI: "
                        f"mean={np.mean(ecg_ac_vals):.3f}, max={np.max(ecg_ac_vals):.3f}"
                    )
                ecg_btb_vals = [v for v in ecg_btb_list if np.isfinite(v)]
                if ecg_btb_vals:
                    print(
                        "  [Debug] ECG beat-to-beat corr SQI: "
                        f"mean={np.mean(ecg_btb_vals):.3f}, max={np.max(ecg_btb_vals):.3f}"
                    )
                ecg_template_vals = [v for v in ecg_template_list if np.isfinite(v)]
                if ecg_template_vals:
                    print(
                        "  [Debug] ECG template-matching SQI: "
                        f"mean={np.mean(ecg_template_vals):.3f}, max={np.max(ecg_template_vals):.3f}"
                    )
                print(
                    "  [Debug] rPPG SNR(dB): "
                    f"mean={np.mean(rppg_snr_list):.3f}, max={np.max(rppg_snr_list):.3f}"
                )
                if rppg_autocorr_list:
                    vals = [v for v in rppg_autocorr_list if np.isfinite(v)]
                    if vals:
                        print(
                            "  [Debug] rPPG autocorr SQI: "
                            f"mean={np.mean(vals):.3f}, max={np.max(vals):.3f}"
                        )
                final_vals = [v for v in rppg_final_list if np.isfinite(v)]
                if final_vals:
                    print(
                        "  [Debug] rPPG fused SQI: "
                        f"mean={np.mean(final_vals):.3f}, max={np.max(final_vals):.3f}"
                    )
                good_count = sum(1 for s in segments_info if s["is_good"])
                print(f"  [Debug] Good segments: {good_count}/{len(segments_info)}")

            blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
            if not blocks:
                print(f"No valid blocks found for {patient_dir}")

            save_blocks = False
            final_ecg_th = self.threshold["ECG"]
            final_rppg_th = self.threshold["rPPG"]

            if self.visualize:
                accepted, final_ecg_th, final_rppg_th = self._visualize_review(
                    timestamps,
                    ecg_signal,
                    rppg_signal,
                    peaks,
                    segments_info,
                    patient_dir,
                    ecg_autocorr_norm_vec,
                    ecg_btb_norm_vec,
                    ecg_template_norm_vec,
                    ecg_fused_vec,
                    rppg_snr_norm_vec,
                    rppg_autocorr_norm_vec,
                    rppg_fused_vec,
                )
                if accepted:
                    for seg in segments_info:
                        seg["is_good"] = (
                            seg["sim_ecg"] >= final_ecg_th and seg["sim_rppg"] >= final_rppg_th
                        )
                    blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
                    save_blocks = True
            elif blocks:
                save_blocks = True

            if save_blocks and blocks:
                self._save_blocks(df, blocks, patient_dir)
                self._save_patient_info(
                    patient_dir,
                    segments_info,
                )

        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            traceback.print_exc()

    def _build_window_quality_vectors(self, ecg_signal, rppg_signal):
        n = len(ecg_signal)
        if n == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        win = max(int(self.sqi_window_sec * self.fs), int(1.5 * self.fs))
        step = max(int(self.sqi_step_sec * self.fs), 1)
        if win >= n:
            ecg_score = ecg_sqi_autocorr(ecg_signal, fs=self.fs)
            ecg_score_btb = ecg_sqi_beat_to_beat_corr(ecg_signal, fs=self.fs)
            rppg_score_snr = compute_snr_db(rppg_signal, fs=self.fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
            rppg_score_autocorr = rppg_sqi_autocorr(rppg_signal, fs=self.fs)
            return (
                np.full(n, ecg_score, dtype=np.float64),
                np.full(n, ecg_score_btb, dtype=np.float64),
                np.full(n, rppg_score_snr, dtype=np.float64),
                np.full(n, rppg_score_autocorr, dtype=np.float64),
            )

        ecg_sum = np.zeros(n, dtype=np.float64)
        ecg_btb_sum = np.zeros(n, dtype=np.float64)
        rppg_snr_sum = np.zeros(n, dtype=np.float64)
        rppg_autocorr_sum = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)

        for start in range(0, n - win + 1, step):
            end = start + win
            ecg_win = ecg_signal[start:end]
            rppg_win = rppg_signal[start:end]

            ecg_score = ecg_sqi_autocorr(ecg_win, fs=self.fs)
            ecg_score_btb = ecg_sqi_beat_to_beat_corr(ecg_win, fs=self.fs)
            rppg_score_snr = compute_snr_db(rppg_win, fs=self.fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
            rppg_score_autocorr = rppg_sqi_autocorr(rppg_win, fs=self.fs)

            ecg_sum[start:end] += ecg_score
            ecg_btb_sum[start:end] += ecg_score_btb
            rppg_snr_sum[start:end] += rppg_score_snr
            rppg_autocorr_sum[start:end] += rppg_score_autocorr
            count[start:end] += 1.0

        # Cover right tail when n-win is not aligned by step.
        last_start = n - win
        if last_start > 0 and ((n - win) % step != 0):
            ecg_win = ecg_signal[last_start:n]
            rppg_win = rppg_signal[last_start:n]
            ecg_score = ecg_sqi_autocorr(ecg_win, fs=self.fs)
            ecg_score_btb = ecg_sqi_beat_to_beat_corr(ecg_win, fs=self.fs)
            rppg_score_snr = compute_snr_db(rppg_win, fs=self.fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
            rppg_score_autocorr = rppg_sqi_autocorr(rppg_win, fs=self.fs)
            ecg_sum[last_start:n] += ecg_score
            ecg_btb_sum[last_start:n] += ecg_score_btb
            rppg_snr_sum[last_start:n] += rppg_score_snr
            rppg_autocorr_sum[last_start:n] += rppg_score_autocorr
            count[last_start:n] += 1.0

        fallback_ecg = ecg_sqi_autocorr(ecg_signal, fs=self.fs)
        fallback_ecg_btb = ecg_sqi_beat_to_beat_corr(ecg_signal, fs=self.fs)
        fallback_rppg_snr = compute_snr_db(rppg_signal, fs=self.fs, lo_hz=0.5, hi_hz=5.0, peak_width_hz=0.15)
        fallback_rppg_autocorr = rppg_sqi_autocorr(rppg_signal, fs=self.fs)

        ecg_vec = np.full(n, fallback_ecg, dtype=np.float64)
        ecg_btb_vec = np.full(n, fallback_ecg_btb, dtype=np.float64)
        rppg_snr_vec = np.full(n, fallback_rppg_snr, dtype=np.float64)
        rppg_autocorr_vec = np.full(n, fallback_rppg_autocorr, dtype=np.float64)

        valid = count > 0
        ecg_vec[valid] = ecg_sum[valid] / count[valid]
        ecg_btb_vec[valid] = ecg_btb_sum[valid] / count[valid]
        rppg_snr_vec[valid] = rppg_snr_sum[valid] / count[valid]
        rppg_autocorr_vec[valid] = rppg_autocorr_sum[valid] / count[valid]

        return ecg_vec, ecg_btb_vec, rppg_snr_vec, rppg_autocorr_vec

    def _build_ecg_template_quality_vector(self, ecg_signal, peaks):
        """Build ECG template-matching SQI vector from beat-by-beat scores."""
        n = len(ecg_signal)
        if n == 0:
            return np.array([])

        fallback = self._max_similarity_to_refs(ecg_signal, self.ecg_refs)
        vec = np.full(n, fallback, dtype=np.float64)

        if peaks is None or len(peaks) < 3:
            return vec

        score_sum = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)

        for i in range(1, len(peaks) - 1):
            peak = peaks[i]
            prev_peak = peaks[i - 1]
            next_peak = peaks[i + 1]

            prev_interval = max(1, peak - prev_peak)
            next_interval = max(1, next_peak - peak)

            start = max(0, int(peak - 0.3 * prev_interval))
            end = min(n, int(peak + 0.7 * next_interval))
            if end - start <= 1:
                continue

            beat_seg = ecg_signal[start:end]
            score = self._max_similarity_to_refs(beat_seg, self.ecg_refs)
            score_sum[start:end] += score
            count[start:end] += 1.0

        valid = count > 0
        vec[valid] = score_sum[valid] / count[valid]
        return np.clip(vec, 0.0, 1.0)

    def _combine_rppg_quality_vectors(self, snr_norm_vec, autocorr_vec):
        parts = []
        weights = []

        if self.rppg_weight_snr > 0 and np.any(np.isfinite(snr_norm_vec)):
            parts.append(np.asarray(snr_norm_vec, dtype=np.float64))
            weights.append(self.rppg_weight_snr)
        if self.rppg_weight_autocorr > 0 and np.any(np.isfinite(autocorr_vec)):
            parts.append(np.asarray(autocorr_vec, dtype=np.float64))
            weights.append(self.rppg_weight_autocorr)

        if not parts:
            return np.zeros_like(np.asarray(snr_norm_vec, dtype=np.float64))

        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

        combined = np.zeros_like(parts[0], dtype=np.float64)
        for i, arr in enumerate(parts):
            valid = np.isfinite(arr)
            if np.any(valid):
                fill = np.nanmedian(arr[valid])
            else:
                fill = 0.0
            arr_filled = np.where(valid, arr, fill)
            combined += w[i] * arr_filled

        return np.clip(combined, 0.0, 1.0)

    def _segment_quality_from_single_vector(self, vec, start, end):
        if end <= start or len(vec) < end:
            return np.nan
        seg = vec[start:end]
        if len(seg) == 0 or np.all(np.isnan(seg)):
            return np.nan
        return float(np.nanpercentile(seg, 25))

    def _read_raw_signal_csv(self, file_path, signal_name):
        expected_columns = RAW_SIGNAL_COLUMNS[signal_name]

        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        normalized_columns = [str(col).strip().lower() for col in df.columns]
        df.columns = normalized_columns
        if all(col in df.columns for col in expected_columns):
            return df[expected_columns].copy()

        try:
            df = pd.read_csv(file_path, header=None)
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            print(f"Error reading headerless {file_path}: {e}")
            return None

        if df.shape[1] < len(expected_columns):
            return None

        df = df.iloc[:, : len(expected_columns)].copy()
        df.columns = expected_columns
        return df

    def _prepare_signal_frame(self, df, value_columns):
        if df is None:
            return None

        numeric_columns = ["timestamp"] + value_columns
        frame = df[numeric_columns].apply(pd.to_numeric, errors="coerce").dropna()
        if frame.empty:
            return None

        frame = frame.sort_values("timestamp")
        frame = frame.drop_duplicates(subset="timestamp", keep="last")
        if len(frame) < 2:
            return None

        return frame.reset_index(drop=True)

    def _resample_signal(self, timestamps, values, new_timestamps):
        if len(timestamps) < 2:
            return np.zeros_like(new_timestamps)

        interpolation_kind = "cubic" if len(timestamps) >= 4 else "linear"
        interpolator = interp1d(
            timestamps,
            values,
            kind=interpolation_kind,
            bounds_error=False,
            fill_value=0.0,
        )
        return interpolator(new_timestamps)

    def _load_patient_dataframe(self, patient_dir):
        ecg_path = os.path.join(patient_dir, "ecg_log.csv")
        rppg_path = os.path.join(patient_dir, "rppg_log.csv")
        ppg_path = os.path.join(patient_dir, "ppg_log.csv")

        ecg_df = self._prepare_signal_frame(self._read_raw_signal_csv(ecg_path, "ecg"), ["ecg"])
        if ecg_df is None:
            print(f"Skipping {patient_dir}: Missing or invalid ecg_log.csv")
            return None

        rppg_df = self._prepare_signal_frame(self._read_raw_signal_csv(rppg_path, "rppg"), ["rppg"])
        if rppg_df is None:
            print(f"Skipping {patient_dir}: Missing or invalid rppg_log.csv")
            return None

        ppg_df = None
        if self.mirror_version == "2":
            ppg_df = self._prepare_signal_frame(
                self._read_raw_signal_csv(ppg_path, "ppg"),
                ["ppg_red", "ppg_ir", "ppg_green"],
            )
            if ppg_df is None:
                print(f"Skipping {patient_dir}: Missing or invalid ppg_log.csv")
                return None

        if abs(rppg_df["timestamp"].iloc[0] - ecg_df["timestamp"].iloc[0]) > 10.0:
            print(f"Skipping {patient_dir}: Large time difference between ECG and rPPG start times")
            return None

        first_timestamp = min(ecg_df["timestamp"].iloc[0], rppg_df["timestamp"].iloc[0])
        last_timestamp = max(ecg_df["timestamp"].iloc[-1], rppg_df["timestamp"].iloc[-1])

        if ppg_df is not None:
            first_timestamp = min(first_timestamp, ppg_df["timestamp"].iloc[0])
            last_timestamp = max(last_timestamp, ppg_df["timestamp"].iloc[-1])

        target_len = int((last_timestamp - first_timestamp) * self.fs)
        if target_len < 2:
            print(f"Skipping {patient_dir}: Not enough samples after raw log alignment")
            return None

        timestamps = np.linspace(first_timestamp, last_timestamp, target_len)
        merged_data = {
            "Timestamp": timestamps,
            "RPPG": self._resample_signal(
                rppg_df["timestamp"].to_numpy(),
                rppg_df["rppg"].to_numpy(),
                timestamps,
            ),
            "ECG": self._resample_signal(
                ecg_df["timestamp"].to_numpy(),
                ecg_df["ecg"].to_numpy(),
                timestamps,
            ),
        }

        if ppg_df is not None:
            ppg_timestamps = ppg_df["timestamp"].to_numpy()
            merged_data["PPG_RED"] = self._resample_signal(
                ppg_timestamps,
                ppg_df["ppg_red"].to_numpy(),
                timestamps,
            )
            merged_data["PPG_IR"] = self._resample_signal(
                ppg_timestamps,
                ppg_df["ppg_ir"].to_numpy(),
                timestamps,
            )
            merged_data["PPG_GREEN"] = self._resample_signal(
                ppg_timestamps,
                ppg_df["ppg_green"].to_numpy(),
                timestamps,
            )

        return pd.DataFrame(merged_data)

    def _detect_peaks(self, ecg_signal):
        if self.ecg_processor:
            self.ecg_processor.process(ecg_signal)
            peaks = self.ecg_processor.get_peaks()
            return np.array(peaks) if peaks is not None else np.array([])

        peaks, _ = signal.find_peaks(ecg_signal, distance=int(self.fs * 0.5))
        return peaks

    def _find_continuous_blocks(self, segments_info, min_len=None):
        if min_len is None:
            min_len = self.segment_length_threshold

        blocks = []
        current_block = []

        for seg in segments_info:
            if seg["is_good"]:
                if not current_block:
                    current_block.append(seg)
                else:
                    last_seg = current_block[-1]
                    if abs(seg["start"] - last_seg["end"]) < 2:
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

    def _visualize_review(
        self,
        timestamps,
        ecg,
        rppg,
        peaks,
        segments_info,
        patient_dir,
        ecg_autocorr_vec=None,
        ecg_btb_vec=None,
        ecg_template_vec=None,
        ecg_fused_vec=None,
        rppg_snr_norm_vec=None,
        rppg_autocorr_vec=None,
        rppg_fused_vec=None,
    ):
        ax1, ax2 = self.axes
        ax1.clear()
        ax2.clear()

        # Recreate ECG SQI twin axis for each refresh to avoid stacked axes.
        if self.ax1_sqi is not None:
            self.ax1_sqi.remove()
            self.ax1_sqi = None
        self.ax1_sqi = ax1.twinx()
        self.ax1_sqi.set_ylim(-0.1, 1.1)
        self.ax1_sqi.set_ylabel("ECG SQI")

        rppg_vis = rppg.copy()
        if len(rppg_vis) > 0:
            r_min = float(np.min(rppg_vis))
            r_max = float(np.max(rppg_vis))
            if r_max > r_min:
                rppg_vis = (rppg_vis - r_min) / (r_max - r_min)

        ax1.plot(timestamps, ecg, label="ECG", color="black", alpha=0.5)
        if ecg_autocorr_vec is not None and len(ecg_autocorr_vec) == len(timestamps):
            self.ax1_sqi.plot(
                timestamps,
                np.nan_to_num(np.clip(ecg_autocorr_vec, 0.0, 1.0), nan=0.0),
                color="tab:blue",
                alpha=0.9,
                linewidth=1.0,
                label="ECG SQI: autocorr",
            )
        if ecg_btb_vec is not None and len(ecg_btb_vec) == len(timestamps):
            self.ax1_sqi.plot(
                timestamps,
                np.nan_to_num(np.clip(ecg_btb_vec, 0.0, 1.0), nan=0.0),
                color="tab:purple",
                alpha=0.85,
                linewidth=1.0,
                label="ECG SQI: btb corr",
            )
        if ecg_template_vec is not None and len(ecg_template_vec) == len(timestamps):
            self.ax1_sqi.plot(
                timestamps,
                np.nan_to_num(np.clip(ecg_template_vec, 0.0, 1.0), nan=0.0),
                color="tab:olive",
                alpha=0.9,
                linewidth=1.0,
                label="ECG SQI: template match",
            )
        if ecg_fused_vec is not None and len(ecg_fused_vec) == len(timestamps):
            self.ax1_sqi.plot(
                timestamps,
                np.nan_to_num(np.clip(ecg_fused_vec, 0.0, 1.0), nan=0.0),
                color="tab:red",
                alpha=0.95,
                linewidth=1.3,
                label="ECG SQI: fused",
            )
        ax2.plot(timestamps, rppg_vis, label="RPPG (Norm)", color="black", alpha=0.5)
        ax2.set_ylim(-0.1, 1.1)

        if rppg_snr_norm_vec is not None and len(rppg_snr_norm_vec) == len(timestamps):
            ax2.plot(
                timestamps,
                np.nan_to_num(rppg_snr_norm_vec, nan=0.0),
                color="tab:orange",
                alpha=0.8,
                linewidth=1.0,
                label="rPPG SQI: SNR",
            )

        if rppg_autocorr_vec is not None and len(rppg_autocorr_vec) == len(timestamps):
            ax2.plot(
                timestamps,
                np.nan_to_num(rppg_autocorr_vec, nan=0.0),
                color="tab:cyan",
                alpha=0.8,
                linewidth=1.0,
                label="rPPG SQI: autocorr",
            )

        if rppg_fused_vec is not None and len(rppg_fused_vec) == len(timestamps):
            ax2.plot(
                timestamps,
                np.nan_to_num(rppg_fused_vec, nan=0.0),
                color="tab:pink",
                alpha=0.95,
                linewidth=1.4,
                label="rPPG SQI: final fused",
            )

        if len(peaks) > 0:
            valid_peaks = peaks[peaks < len(timestamps)]
            ax1.plot(timestamps[valid_peaks], ecg[valid_peaks], "r.", markersize=5, label="Peaks")

        rppg_peaks = find_rppg_peaks(rppg, fs=self.fs)
        if len(rppg_peaks) > 0:
            valid_rppg_peaks = rppg_peaks[rppg_peaks < len(timestamps)]
            ax2.plot(
                timestamps[valid_rppg_peaks],
                rppg_vis[valid_rppg_peaks],
                "r.",
                markersize=5,
                label="Peaks",
            )

        for seg in segments_info:
            t_start = timestamps[seg["start"]]
            ax1.axvline(x=t_start, color="blue", alpha=0.05, linestyle="-", linewidth=0.5)

        patient_name = os.path.basename(patient_dir)
        ax1.set_title(
            "Patient: "
            f"{patient_name} - ECG | weights AC={self.ecg_weight_autocorr:.2f}, "
            f"BTB={self.ecg_weight_btb_corr:.2f}, TM={self.ecg_weight_template:.2f}"
        )
        ax2.set_title(
            "RPPG | SQI shown: SNR + autocorr + fused "
            f"(weights SNR={self.rppg_weight_snr:.2f}, AC={self.rppg_weight_autocorr:.2f})"
        )

        h1, l1 = ax1.get_legend_handles_labels()
        h1s, l1s = self.ax1_sqi.get_legend_handles_labels()
        if h1s:
            ax1.legend(h1 + h1s, l1 + l1s, loc="upper right")
        else:
            ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        plt.suptitle("Adjust thresholds. Press 'y' to accept, 'n' to reject.")

        self.ax_ecg_thresh.clear()
        self.ax_rppg_thresh.clear()
        self.slider_ecg = Slider(self.ax_ecg_thresh, "ECG Thresh", 0.0, 1.0, valinit=self.threshold["ECG"])
        self.slider_rppg = Slider(
            self.ax_rppg_thresh,
            "RPPG Fused Thresh",
            0.0,
            1.0,
            valinit=self.threshold["rPPG"],
        )

        self.current_spans = []

        def update(_):
            for span in self.current_spans:
                span.remove()
            self.current_spans = []

            ecg_th = self.slider_ecg.val
            rppg_th = self.slider_rppg.val

            for seg in segments_info:
                seg["is_good"] = (seg["sim_ecg"] >= ecg_th) and (seg["sim_rppg"] >= rppg_th)
                t_start = timestamps[seg["start"]]
                t_end = timestamps[seg["end"]]

                if seg["sim_ecg"] >= ecg_th:
                    self.current_spans.append(ax1.axvspan(t_start, t_end, color="yellow", alpha=0.2))
                if seg["sim_rppg"] >= rppg_th:
                    self.current_spans.append(ax2.axvspan(t_start, t_end, color="yellow", alpha=0.2))

            current_blocks = self._find_continuous_blocks(segments_info, min_len=self.segment_length_threshold)
            for block in current_blocks:
                start_idx = block[0]["start"]
                end_idx = block[-1]["end"]
                t_start = timestamps[start_idx]
                t_end = timestamps[end_idx]
                self.current_spans.append(ax1.axvspan(t_start, t_end, color="green", alpha=0.4))
                self.current_spans.append(ax2.axvspan(t_start, t_end, color="green", alpha=0.4))

            self.fig.canvas.draw_idle()

        self.slider_ecg.on_changed(update)
        self.slider_rppg.on_changed(update)
        update(None)

        self.last_key = None
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        try:
            while True:
                plt.pause(0.1)
                if self.last_key == "y":
                    return True, self.slider_ecg.val, self.slider_rppg.val
                if self.last_key == "n":
                    return False, self.threshold["ECG"], self.threshold["rPPG"]
        except Exception:
            return False, self.threshold["ECG"], self.threshold["rPPG"]

    def _save_patient_info(
        self,
        patient_dir,
        segments_info,
    ):
        patient_id = os.path.basename(patient_dir)
        lab_patient_id = int(patient_id.replace("patient_", ""))
        good_segments = [seg for seg in segments_info if seg["is_good"]]
        if not good_segments:
            return

        ecg_sqi_avg = np.mean([seg["sim_ecg"] for seg in good_segments])
        ecg_sqi_autocorr_avg = np.mean([seg.get("sim_ecg_autocorr", np.nan) for seg in good_segments])
        ecg_sqi_btb_corr_avg = np.mean([seg.get("sim_ecg_btb_corr", np.nan) for seg in good_segments])
        ecg_sqi_template_avg = np.mean([seg.get("sim_ecg_template", np.nan) for seg in good_segments])
        rppg_sqi_avg = np.mean([seg["sim_rppg"] for seg in good_segments])
        rppg_snr_db_avg = np.mean([seg.get("sim_rppg_snr_db", np.nan) for seg in good_segments])
        rppg_snr_norm_avg = np.mean([seg.get("sim_rppg_snr_norm", np.nan) for seg in good_segments])
        rppg_sqi_autocorr_avg = np.mean([seg.get("sim_rppg_autocorr", np.nan) for seg in good_segments])

        patient_info = self.patient_info_lookup.get(lab_patient_id, {})
        hospital_patient_id = patient_info.get("hospital_patient_id", "")
        low_bp = patient_info.get("low_blood_pressure", -1)
        high_bp = patient_info.get("high_blood_pressure", -1)
        if hospital_patient_id == "":
            print(f"  [Warning] No hospital_patient_id found for lab_patient_id {lab_patient_id}")

        self.cleaned_patient_info.append(
            {
                "Lab_Patient_ID": lab_patient_id,
                "Hospital_Patient_ID": hospital_patient_id,
                "ECG_SQI_AVG": ecg_sqi_avg,
                "ECG_SQI_AUTOCORR_AVG": ecg_sqi_autocorr_avg,
                "ECG_SQI_BTB_CORR_AVG": ecg_sqi_btb_corr_avg,
                "ECG_SQI_TEMPLATE_AVG": ecg_sqi_template_avg,
                "rPPG_SQI_AVG": rppg_sqi_avg,
                "rPPG_SQI_SNR_DB_AVG": rppg_snr_db_avg,
                "rPPG_SQI_SNR_NORM_AVG": rppg_snr_norm_avg,
                "rPPG_SQI_AUTOCORR_AVG": rppg_sqi_autocorr_avg,
                "Low_Blood_Pressure": low_bp,
                "High_Blood_Pressure": high_bp,
            }
        )

    def _save_blocks(self, df, blocks, patient_dir):
        patient_id = os.path.basename(patient_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for i, block in enumerate(blocks):
            start_idx = block[0]["start"]
            end_idx = block[-1]["end"]

            block_df = df.iloc[start_idx:end_idx][["Timestamp", "RPPG", "ECG"]].copy()

            if "ECG" in block_df.columns and self.polarity == "neg":
                block_df["ECG"] = -block_df["ECG"]

            for col in ["RPPG", "ECG"]:
                if col in block_df.columns:
                    block_df[col] = (block_df[col] - block_df[col].mean()) / block_df[col].std()

            filename = f"{patient_id}_{i + 1}.csv"
            filepath = os.path.join(self.output_dir, filename)
            block_df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")


if __name__ == "__main__":
    mirror_id = 6
    parser = argparse.ArgumentParser(
        description="Auto Wash Patient Data with ECG (autocorr/btb/template) SQI + rPPG SNR/autocorr SQI"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=f"./mirror{mirror_id}_data",
        help="Directory containing patient folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"./mirror{mirror_id}_auto_cleaned_sqi",
        help="Directory to save cleaned segments",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="./reference_signals",
        help="Directory containing ECG template reference signals",
    )
    parser.add_argument(
        "--patient_info_csv",
        type=str,
        default=f"./merged_patient_info_{mirror_id}.csv",
        help="CSV file with patient info including blood pressure",
    )
    parser.add_argument(
        "--mirror_version",
        type=str,
        default="1",
        choices=["1", "2"],
        help="Mirror data version: 1 for mirror1/2, 2 for mirror4/5/6",
    )
    parser.add_argument("--threshold_ecg", type=float, default=0.4, help="Final fused ECG SQI threshold")
    parser.add_argument("--threshold_rppg", type=float, default=0.4, help="Final fused rPPG SQI threshold in [0,1]")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization and manual review")
    parser.add_argument("--polarity", type=str, default="neg", choices=["neg", "pos"], help="ECG signal polarity: 'neg' or 'pos'")
    parser.add_argument(
        "--rppg_weight_snr",
        type=float,
        default=0.2,
        help="Weight of normalized SNR in final fused rPPG SQI",
    )
    parser.add_argument(
        "--rppg_weight_autocorr",
        type=float,
        default=0.8,
        help="Weight of autocorr SQI in final fused rPPG SQI",
    )
    parser.add_argument(
        "--ecg_weight_autocorr",
        type=float,
        default=0.4,
        help="Weight of ECG autocorr SQI in final fused ECG SQI",
    )
    parser.add_argument(
        "--ecg_weight_btb_corr",
        type=float,
        default=0.3,
        help="Weight of ECG beat-to-beat correlation SQI in final fused ECG SQI",
    )
    parser.add_argument(
        "--ecg_weight_template",
        type=float,
        default=0.3,
        help="Weight of ECG template-matching SQI in final fused ECG SQI",
    )

    args = parser.parse_args()

    washer = AutoWasherSQI(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        reference_dir=args.reference_dir,
        patient_info_csv=args.patient_info_csv,
        threshold={"ECG": args.threshold_ecg, "rPPG": args.threshold_rppg},
        visualize=args.visualize,
        mirror_version=args.mirror_version,
        rppg_weight_snr=args.rppg_weight_snr,
        rppg_weight_autocorr=args.rppg_weight_autocorr,
        ecg_weight_autocorr=args.ecg_weight_autocorr,
        ecg_weight_btb_corr=args.ecg_weight_btb_corr,
        ecg_weight_template=args.ecg_weight_template,
        polarity=args.polarity,
    )
    washer.process_all()