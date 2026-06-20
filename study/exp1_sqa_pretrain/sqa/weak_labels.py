"""Conservative ECG weak labels for multi-task signal quality assessment."""

from dataclasses import asdict, dataclass
import glob
import os

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass(frozen=True)
class WeakLabelConfig:
    """All tunable weak-label thresholds in one place."""

    min_hr_bpm: float = 30.0
    max_hr_bpm: float = 220.0
    match_tolerance_sec: float = 0.08
    refine_radius_sec: float = 0.05
    filter_low_hz: float = 5.0
    filter_high_hz: float = 20.0
    energy_window_sec: float = 0.12
    beat_pre_sec: float = 0.25
    beat_post_sec: float = 0.40
    min_template_beats: int = 3
    bad_threshold: float = 0.75
    flatline_std: float = 1e-4
    small_robust_amplitude: float = 1e-3
    clipping_fraction: float = 0.10
    impulse_ratio: float = 12.0

    def to_dict(self):
        return asdict(self)


FEATURE_NAMES = (
    "rpeak_agreement_score",
    "rr_plausibility_score",
    "template_corr_score",
    "bad_segment_score",
    "autocorr_score",
)


def _robust_scale(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median, max(1.4826 * mad, 1e-8)


def _sanitize_signal(ecg):
    x = np.asarray(ecg, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x), 1.0
    missing_fraction = 1.0 - float(finite.mean())
    if not finite.all():
        indices = np.arange(len(x))
        x[~finite] = np.interp(indices[~finite], indices[finite], x[finite])
    return x, missing_fraction


def _bandpass(ecg, fs, config):
    nyquist = 0.5 * fs
    low = max(0.5, config.filter_low_hz)
    high = min(config.filter_high_hz, 0.90 * nyquist)
    if high <= low or len(ecg) < 16:
        return ecg - np.mean(ecg)
    sos = butter(2, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, ecg)
    except ValueError:
        return ecg - np.mean(ecg)


def _refine_peaks(peaks, reference, radius):
    refined = []
    for peak in peaks:
        left = max(0, int(peak) - radius)
        right = min(len(reference), int(peak) + radius + 1)
        if right > left:
            refined.append(left + int(np.argmax(np.abs(reference[left:right]))))
    return np.unique(refined).astype(np.int64)


def _detect_amplitude_peaks(filtered, fs, config):
    magnitude = np.abs(filtered)
    median, scale = _robust_scale(magnitude)
    distance = max(1, int(round(fs * 60.0 / config.max_hr_bpm)))
    peaks, _ = find_peaks(
        magnitude,
        distance=distance,
        height=median + 2.0 * scale,
        prominence=0.75 * scale,
    )
    radius = max(1, int(round(config.refine_radius_sec * fs)))
    return _refine_peaks(peaks, filtered, radius)


def _detect_energy_peaks(filtered, fs, config):
    energy = np.gradient(filtered) ** 2
    window = max(1, int(round(config.energy_window_sec * fs)))
    envelope = np.convolve(energy, np.ones(window) / window, mode="same")
    median, scale = _robust_scale(envelope)
    distance = max(1, int(round(fs * 60.0 / config.max_hr_bpm)))
    peaks, _ = find_peaks(
        envelope,
        distance=distance,
        height=median + 1.5 * scale,
        prominence=0.5 * scale,
    )
    radius = max(1, int(round(config.refine_radius_sec * fs)))
    return _refine_peaks(peaks, filtered, radius)


def _match_peaks(peaks_a, peaks_b, tolerance):
    """One-to-one peak matching and a symmetric agreement score."""
    i = j = 0
    matched = []
    while i < len(peaks_a) and j < len(peaks_b):
        delta = int(peaks_a[i]) - int(peaks_b[j])
        if abs(delta) <= tolerance:
            matched.append(int(round((int(peaks_a[i]) + int(peaks_b[j])) / 2)))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1

    denominator = len(peaks_a) + len(peaks_b)
    score = 2.0 * len(matched) / denominator if denominator else 0.0
    return np.asarray(matched, dtype=np.int64), float(np.clip(score, 0.0, 1.0))


def _rr_plausibility(peaks, fs, config):
    if len(peaks) < 2:
        return 0.0
    rr = np.diff(peaks) / float(fs)
    rr_min = 60.0 / config.max_hr_bpm
    rr_max = 60.0 / config.min_hr_bpm
    interval_score = float(np.mean((rr >= rr_min) & (rr <= rr_max)))

    median_rr = float(np.median(rr))
    median_hr = 60.0 / max(median_rr, 1e-8)
    hr_score = float(config.min_hr_bpm <= median_hr <= config.max_hr_bpm)

    if len(rr) < 2:
        jump_score = 0.75
    else:
        relative_jumps = np.abs(np.diff(rr)) / np.maximum(rr[:-1], rr[1:])
        # Broad tolerance preserves genuine rhythm variability.
        jump_score = float(np.mean(relative_jumps <= 0.60))

    return float(np.clip(0.50 * interval_score + 0.25 * hr_score + 0.25 * jump_score, 0.0, 1.0))


def _normalize_beat(beat):
    beat = np.asarray(beat, dtype=np.float64)
    finite = np.isfinite(beat)
    if finite.sum() < 3:
        return None
    centered = beat.copy()
    mean = np.mean(centered[finite])
    std = np.std(centered[finite])
    if std <= 1e-8:
        return None
    centered[finite] = (centered[finite] - mean) / std
    return centered


def _template_correlation(ecg, peaks, fs, config, reference_template=None):
    pre = int(round(config.beat_pre_sec * fs))
    post = int(round(config.beat_post_sec * fs))
    beats = [
        _normalize_beat(ecg[peak - pre:peak + post])
        for peak in peaks
        if peak - pre >= 0 and peak + post <= len(ecg)
    ]
    beats = [beat for beat in beats if beat is not None]
    if len(beats) < config.min_template_beats:
        return 0.0, len(beats)

    beat_array = np.stack(beats)
    if reference_template is None:
        template = _normalize_beat(np.median(beat_array, axis=0))
        polarity_invariant = False
    else:
        if len(reference_template) != beat_array.shape[1]:
            raise ValueError("Reference template length does not match extracted beats.")
        template = _normalize_beat(reference_template)
        polarity_invariant = True

    if template is None:
        return 0.0, len(beats)
    valid = np.isfinite(template)
    if valid.sum() < 3:
        return 0.0, len(beats)

    correlations = []
    for beat in beat_array:
        beat_valid = valid & np.isfinite(beat)
        if beat_valid.sum() < 3:
            continue
        correlation = float(np.corrcoef(beat[beat_valid], template[beat_valid])[0, 1])
        correlations.append(abs(correlation) if polarity_invariant else correlation)

    score = np.median(correlations) if correlations else 0.0
    return float(np.clip(score, 0.0, 1.0)), len(beats)


def _load_reference_template(reference_dir, target_fs, config):
    """Build an R-aligned median beat from reference_ecg CSV files."""
    paths = sorted(glob.glob(os.path.join(reference_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV reference ECG files found in: {reference_dir}")

    pre = int(round(config.beat_pre_sec * target_fs))
    post = int(round(config.beat_post_sec * target_fs))
    target_times = np.arange(-pre, post, dtype=np.float64) / target_fs
    aligned_beats = []

    for path in paths:
        try:
            data = np.genfromtxt(path, delimiter=",", names=True)
            column_names = {name.lower(): name for name in (data.dtype.names or ())}
            time_name = column_names.get("timestamps") or column_names.get("timestamp")
            ecg_name = column_names.get("ecg")
            if time_name is None or ecg_name is None:
                continue

            timestamps = np.asarray(data[time_name], dtype=np.float64)
            ecg = np.asarray(data[ecg_name], dtype=np.float64)
            finite = np.isfinite(timestamps) & np.isfinite(ecg)
            timestamps, ecg = timestamps[finite], ecg[finite]
            if len(ecg) < 16:
                continue

            dt = float(np.median(np.diff(timestamps)))
            if dt <= 0:
                continue
            reference_fs = 1.0 / dt
            filtered = _bandpass(ecg, reference_fs, config)
            r_peak = int(np.argmax(np.abs(filtered)))
            relative_times = timestamps - timestamps[r_peak]

            aligned = np.interp(
                target_times, relative_times, ecg, left=np.nan, right=np.nan
            )
            aligned = _normalize_beat(aligned)
            if aligned is not None:
                # Reference polarity must not depend on recording lead direction.
                peak_index = int(np.argmin(np.abs(target_times)))
                if np.isfinite(aligned[peak_index]) and aligned[peak_index] < 0:
                    aligned = -aligned
                aligned_beats.append(aligned)
        except (OSError, ValueError, TypeError):
            continue

    if not aligned_beats:
        raise ValueError(f"No usable reference ECG templates found in: {reference_dir}")

    reference_stack = np.stack(aligned_beats)
    template = np.full(reference_stack.shape[1], np.nan, dtype=np.float64)
    for index in range(reference_stack.shape[1]):
        values = reference_stack[:, index]
        values = values[np.isfinite(values)]
        if len(values):
            template[index] = np.median(values)

    template = _normalize_beat(template)
    if template is None:
        raise ValueError(f"Reference ECG template is degenerate: {reference_dir}")
    return template


def _bad_segment_score(raw_ecg, missing_fraction, config):
    if len(raw_ecg) < 2:
        return 1.0

    finite = raw_ecg[np.isfinite(raw_ecg)]
    if len(finite) < 2:
        return 1.0

    std = float(np.std(finite))
    low, high = np.percentile(finite, [5.0, 95.0])
    robust_amplitude = float(high - low)
    amplitude_scale = max(robust_amplitude, 1e-8)

    flat_score = max(
        np.clip((config.flatline_std - std) / config.flatline_std, 0.0, 1.0),
        np.clip(
            (config.small_robust_amplitude - robust_amplitude)
            / config.small_robust_amplitude,
            0.0,
            1.0,
        ),
    )

    minimum, maximum = float(np.min(finite)), float(np.max(finite))
    clip_tolerance = max(1e-10, amplitude_scale * 1e-5)
    edge_fraction = max(
        float(np.mean(np.abs(finite - minimum) <= clip_tolerance)),
        float(np.mean(np.abs(finite - maximum) <= clip_tolerance)),
    )
    clipping_score = float(
        np.clip(
            (edge_fraction - config.clipping_fraction)
            / max(1e-8, 0.50 - config.clipping_fraction),
            0.0,
            1.0,
        )
    )

    differences = np.abs(np.diff(finite))
    _, diff_scale = _robust_scale(differences)
    derivative_ratio = (
        float(np.max(differences) / diff_scale) if len(differences) else 0.0
    )
    centered = np.abs(finite - np.median(finite))
    amplitude_ratio = float(np.max(centered) / amplitude_scale)
    # A physiological QRS can have a very steep derivative. Require an extreme
    # amplitude outlier as well before treating the event as an impulse artifact.
    derivative_excess = np.clip(
        (derivative_ratio - config.impulse_ratio) / config.impulse_ratio, 0.0, 1.0
    )
    amplitude_excess = np.clip((amplitude_ratio - 4.0) / 4.0, 0.0, 1.0)
    impulse_score = float(derivative_excess * amplitude_excess)
    missing_score = float(np.clip(missing_fraction / 0.02, 0.0, 1.0))

    return float(np.clip(max(flat_score, clipping_score, impulse_score, missing_score), 0.0, 1.0))


def _autocorr_score(filtered, fs, config):
    if len(filtered) < 4 or np.std(filtered) <= 1e-8:
        return 0.0

    energy = np.gradient(filtered) ** 2
    window = max(1, int(round(config.energy_window_sec * fs)))
    envelope = np.convolve(energy, np.ones(window) / window, mode="same")
    envelope = envelope - np.mean(envelope)
    acf = np.correlate(envelope, envelope, mode="full")[len(envelope) - 1:]
    if acf[0] <= 1e-12:
        return 0.0
    acf /= acf[0]

    lag_min = max(1, int(round(fs * 60.0 / config.max_hr_bpm)))
    lag_max = min(len(acf) - 1, int(round(fs * 60.0 / config.min_hr_bpm)))
    if lag_max <= lag_min:
        return 0.0
    return float(np.clip(np.max(acf[lag_min:lag_max + 1]), 0.0, 1.0))


class ECGWeakLabelGenerator:
    """Generate soft QRS and morphology reliability labels from ECG heuristics."""

    def __init__(self, config=None, template_source="window", reference_dir=None):
        self.config = config or WeakLabelConfig()
        if template_source not in {"window", "reference"}:
            raise ValueError("template_source must be 'window' or 'reference'")
        if template_source == "reference" and not reference_dir:
            raise ValueError("reference_dir is required when template_source='reference'")
        self.template_source = template_source
        self.reference_dir = reference_dir
        self._reference_templates = {}

    def _reference_template(self, sampling_rate_hz):
        if self.template_source == "window":
            return None
        cache_key = float(sampling_rate_hz)
        if cache_key not in self._reference_templates:
            self._reference_templates[cache_key] = _load_reference_template(
                self.reference_dir, sampling_rate_hz, self.config
            )
        return self._reference_templates[cache_key]

    def __call__(self, ecg, sampling_rate_hz):
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")

        raw = np.asarray(ecg, dtype=np.float64).reshape(-1)
        sanitized, missing_fraction = _sanitize_signal(raw)
        bad_score = _bad_segment_score(raw, missing_fraction, self.config)
        filtered = _bandpass(sanitized, sampling_rate_hz, self.config)

        peaks_a = _detect_amplitude_peaks(filtered, sampling_rate_hz, self.config)
        peaks_b = _detect_energy_peaks(filtered, sampling_rate_hz, self.config)
        tolerance = max(1, int(round(self.config.match_tolerance_sec * sampling_rate_hz)))
        consensus_peaks, agreement = _match_peaks(peaks_a, peaks_b, tolerance)

        rr_score = _rr_plausibility(consensus_peaks, sampling_rate_hz, self.config)
        template_score, valid_beats = _template_correlation(
            sanitized,
            consensus_peaks,
            sampling_rate_hz,
            self.config,
            reference_template=self._reference_template(sampling_rate_hz),
        )
        autocorr_score = _autocorr_score(filtered, sampling_rate_hz, self.config)

        y_qrs = (
            0.40 * agreement
            + 0.25 * rr_score
            + 0.20 * autocorr_score
            + 0.15 * (1.0 - bad_score)
        )
        y_morph = (
            0.45 * template_score
            + 0.20 * agreement
            + 0.15 * rr_score
            + 0.20 * (1.0 - bad_score)
        )

        if bad_score >= self.config.bad_threshold:
            y_qrs = min(y_qrs, 0.10)
            y_morph = min(y_morph, 0.05)

        evidence = np.asarray([agreement, rr_score, autocorr_score], dtype=np.float64)
        consistency = float(np.clip(1.0 - 2.0 * np.std(evidence), 0.0, 1.0))
        detector_confidence = float(min(len(peaks_a), len(peaks_b)) >= 2)
        beat_confidence = float(
            min(1.0, valid_beats / max(1, self.config.min_template_beats))
        )
        sample_weight = 0.30 + 0.35 * consistency + 0.20 * detector_confidence + 0.15 * beat_confidence
        if bad_score >= self.config.bad_threshold:
            sample_weight = max(sample_weight, 0.90)
        elif len(consensus_peaks) < 2:
            sample_weight *= 0.60
        sample_weight = float(np.clip(sample_weight, 0.10, 1.0))

        features = {
            "rpeak_agreement_score": float(agreement),
            "rr_plausibility_score": float(rr_score),
            "template_corr_score": float(template_score),
            "bad_segment_score": float(bad_score),
            "autocorr_score": float(autocorr_score),
            "detector_a_peaks": int(len(peaks_a)),
            "detector_b_peaks": int(len(peaks_b)),
            "consensus_peaks": int(len(consensus_peaks)),
            "valid_template_beats": int(valid_beats),
        }
        result = {
            "y_qrs": float(np.clip(y_qrs, 0.0, 1.0)),
            "y_morph": float(np.clip(y_morph, 0.0, 1.0)),
            "sample_weight": sample_weight,
            "features": features,
        }

        assert 0.0 <= result["y_qrs"] <= 1.0
        assert 0.0 <= result["y_morph"] <= 1.0
        assert 0.0 <= result["sample_weight"] <= 1.0
        assert all(np.isfinite(value) for value in result.values() if isinstance(value, float))
        return result
