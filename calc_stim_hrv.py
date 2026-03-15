"""
For stimulated patients in mirror4 (same hospital_patient_id, same day,
2+ sessions), calculate HRV RMSSD before and after electrical stimulation
using ECG data.

Within each (hospital_patient_id, date) group, sessions are sorted by
timestamp. The first session is "before" stimulation, subsequent sessions
are "after" stimulation.

Interactive visualization mode (--vis):
  - Left/Right arrow keys: navigate between groups
  - Click + drag on any subplot: select a segment (yellow highlight)
  - Hold Shift + drag: add additional segments
  - Press 'r': reset all segment selections
  - Press 'c': recalculate & print RMSSD from selected segments
  - Press 'q': quit
"""

import os
import re
import csv
import json
import sys
import argparse
import numpy as np
from scipy import signal as sig
from collections import defaultdict
from datetime import datetime

from ecg.ecg_process import ECGProcess

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.patches import Rectangle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIRROR4_DIR = os.path.join(BASE_DIR, "mirror4_data")
MERGED_CSV = os.path.join(BASE_DIR, "merged_patient_info_4.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "stim_hrv_rmssd_mirror4.csv")

FS = 512  # ECG sampling rate (Hz)

PATIENT_ID_RE = re.compile(r"^Patient ID:\s*(\S+)", re.MULTILINE)
TIMESTAMP_RE = re.compile(r"^Session Timestamp:\s*(.+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

# Shared Pan-Tompkins ECG processor
_ecg_processor = ECGProcess(method='pt', fs=FS)


def detect_r_peaks(ecg_signal, fs=FS):
    """Detect R-peaks using Pan-Tompkins algorithm (same as auto_wash.py).

    Returns
    -------
    peaks : np.ndarray
        Indices of detected R-peaks in *ecg_signal*.
    pantompkins : np.ndarray
        The Pan-Tompkins integrated signal (for visualisation).
    """
    _ecg_processor.fs = fs
    _ecg_processor.process(ecg_signal)
    peaks = _ecg_processor.get_peaks()
    additional = _ecg_processor.get_additional_signals() or {}
    pantompkins = additional.get('pantompkins', np.array([]))
    return (np.array(peaks) if peaks is not None else np.array([])), pantompkins


def compute_rmssd(timestamps, peaks):
    """
    Compute RMSSD from detected R-peaks.
    Returns RMSSD in milliseconds, or None if insufficient data.
    """
    if len(peaks) < 3:
        return None

    peak_times = timestamps[peaks]
    rr_intervals = np.diff(peak_times)  # in seconds

    # Keep physiologically plausible RR intervals (30–200 BPM → 0.3–2.0 s)
    valid_mask = (rr_intervals > 0.3) & (rr_intervals < 2.0)
    valid_rr = rr_intervals[valid_mask]

    if len(valid_rr) < 3:
        return None

    successive_diff = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(successive_diff ** 2)) * 1000  # ms
    return rmssd


def compute_hr(timestamps, peaks):
    """Compute mean heart rate from R-peaks. Returns BPM or None."""
    if len(peaks) < 3:
        return None
    peak_times = timestamps[peaks]
    rr = np.diff(peak_times)
    valid_rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(valid_rr) < 2:
        return None
    return 60.0 / np.mean(valid_rr)


def compute_mean_rr(timestamps, peaks):
    """Compute mean RR interval in milliseconds from R-peaks."""
    if len(peaks) < 3:
        return None
    peak_times = timestamps[peaks]
    rr = np.diff(peak_times)
    valid_rr = rr[(rr > 0.3) & (rr < 2.0)]
    if len(valid_rr) < 2:
        return None
    return np.mean(valid_rr) * 1000  # ms


def load_sbp_map(csv_path):
    """Load SBP (high_blood_pressure) from merged_patient_info CSV.

    Returns dict: lab_patient_id (int) -> SBP (float or None).
    """
    sbp_map = {}
    if not os.path.isfile(csv_path):
        print(f"Warning: {csv_path} not found, SBP data unavailable.")
        return sbp_map
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["lab_patient_id"])
                sbp = float(row["high_blood_pressure"])
                if sbp <= 0:
                    sbp = None
            except (ValueError, KeyError):
                continue
            sbp_map[pid] = sbp
    return sbp_map


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def read_patient_info(patient_dir):
    """Read patient_info.txt, return (patient_id, hospital_patient_id, session_timestamp_str)."""
    info_path = os.path.join(patient_dir, "patient_info.txt")
    if not os.path.isfile(info_path):
        return None, None, None

    content = None
    for enc in ("utf-8", "gbk"):
        try:
            with open(info_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if content is None:
        return None, None, None

    m = PATIENT_ID_RE.search(content)
    patient_id = m.group(1) if m else None

    # hospital_patient_id from JSON
    hospital_patient_id = None
    match = re.search(r'Patient Info:\s*"?(.*)"?\s*$', content, re.MULTILINE)
    if match:
        raw = match.group(1).strip().strip('"').replace('\\"', '"')
        try:
            info = json.loads(raw)
            hospital_patient_id = info.get("patient_id", None)
        except (json.JSONDecodeError, TypeError):
            pass

    # session timestamp
    session_ts = None
    tm = TIMESTAMP_RE.search(content)
    if tm:
        session_ts = tm.group(1).strip()

    return patient_id, hospital_patient_id, session_ts


def read_ecg(patient_dir):
    """Read ecg_log.csv -> (timestamps, ecg_values) as numpy arrays."""
    ecg_path = os.path.join(patient_dir, "ecg_log.csv")
    if not os.path.isfile(ecg_path):
        return None, None
    timestamps = []
    values = []
    with open(ecg_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                timestamps.append(float(row[0]))
                values.append(-float(row[1]))
            except (ValueError, IndexError):
                continue
    if len(timestamps) < 100:
        return None, None
    return np.array(timestamps), np.array(values)


# ---------------------------------------------------------------------------
# Interactive Visualization
# ---------------------------------------------------------------------------

class StimHRVViewer:
    """Interactive ECG viewer for before/after stimulation groups."""

    def __init__(self, stim_groups_data):
        """
        stim_groups_data: list of dicts, each:
          {
            'key': (hosp_id_int, date),
            'sessions': [
              {'rec': {...}, 'ts': np.array, 'ecg': np.array,
               'pantompkins': np.array, 'peaks': np.array, 'phase': str,
               'rmssd': float|None, 'hr': float|None},
              ...
            ]
          }
        """
        self.groups = stim_groups_data
        self.current_idx = 0

        # Per-subplot segment selections: list of lists of (xmin, xmax)
        # indexed by subplot position
        self.selections = {}  # ax_index -> [(xmin, xmax), ...]
        self.highlight_patches = {}  # ax_index -> [Rectangle, ...]
        self.shift_held = False

        self._build_figure()
        self._draw_group()
        plt.show()

    def _build_figure(self):
        max_sessions = max(len(g['sessions']) for g in self.groups)
        self.n_rows = max_sessions
        self.fig, self.axes_arr = plt.subplots(
            max_sessions, 1, figsize=(16, 4 * max_sessions), squeeze=False
        )
        self.axes = [self.axes_arr[i, 0] for i in range(max_sessions)]

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)

        # Build span selectors for each axis
        self.span_selectors = []
        for i, ax in enumerate(self.axes):
            ss = SpanSelector(
                ax, lambda xmin, xmax, idx=i: self._on_span(idx, xmin, xmax),
                'horizontal', useblit=True,
                props=dict(alpha=0.3, facecolor='yellow'),
                interactive=False,
                button=[1],
            )
            self.span_selectors.append(ss)

        self.fig.subplots_adjust(hspace=0.35, top=0.93, bottom=0.05,
                                 left=0.06, right=0.98)

    def _clear_selections(self):
        for idx in list(self.highlight_patches.keys()):
            for p in self.highlight_patches[idx]:
                p.remove()
        self.selections.clear()
        self.highlight_patches.clear()

    def _draw_group(self):
        group = self.groups[self.current_idx]
        sessions = group['sessions']
        key = group['key']

        self._clear_selections()

        for i, ax in enumerate(self.axes):
            ax.clear()
            if i < len(sessions):
                sess = sessions[i]
                ts = sess['ts']
                ecg = sess['ecg']
                pantompkins = sess['pantompkins']
                peaks = sess['peaks']
                phase = sess['phase']
                rec = sess['rec']
                rmssd = sess['rmssd']
                hr = sess['hr']
                mean_rr = sess.get('mean_rr')
                sbp = sess.get('sbp')

                # Use relative time (seconds from start)
                t_rel = ts - ts[0]

                ax.plot(t_rel, ecg, color='steelblue', linewidth=0.6,
                        label='ECG')
                if len(peaks) > 0:
                    ax.plot(t_rel[peaks], ecg[peaks], 'rv',
                            markersize=6, label=f'R-peaks ({len(peaks)})')

                # Show Pan-Tompkins signal on twin y-axis
                if len(pantompkins) > 0:
                    ax2 = ax.twinx()
                    ax2.plot(t_rel, pantompkins, color='orange', linewidth=0.4,
                             alpha=0.5, label='Pan-Tompkins')
                    ax2.set_ylabel('PT amplitude', color='orange', fontsize=8)
                    ax2.tick_params(axis='y', labelcolor='orange', labelsize=7)
                    ax2.set_ylim(bottom=0)

                rmssd_s = f"{rmssd:.2f}" if rmssd is not None else "N/A"
                hr_s = f"{hr:.1f}" if hr is not None else "N/A"
                rr_s = f"{mean_rr:.1f}" if mean_rr is not None else "N/A"
                sbp_s = f"{sbp:.0f}" if sbp is not None else "N/A"

                phase_color = '#2e7d32' if phase == 'before' else '#c62828'
                phase_label = phase.upper()

                ax.set_title(
                    f"[{phase_label}]  {rec['patient_folder']}  |  "
                    f"{rec['session_date']} {rec['session_time']}  |  "
                    f"RMSSD={rmssd_s}  meanRR={rr_s}  HR={hr_s}  SBP={sbp_s}",
                    fontsize=11, fontweight='bold', color=phase_color
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ECG amplitude")
                ax.legend(loc='upper right', fontsize=8)
                ax.set_xlim(t_rel[0], t_rel[-1])

                self.span_selectors[i].set_active(True)
                self.selections[i] = []
                self.highlight_patches[i] = []
            else:
                ax.set_visible(False)
                self.span_selectors[i].set_active(False)

        # Super title
        hosp_id = sessions[0]['rec']['hospital_patient_id']
        date = sessions[0]['rec']['session_date']
        self.fig.suptitle(
            f"Group {self.current_idx + 1}/{len(self.groups)}  |  "
            f"Hospital ID: {hosp_id}  |  Date: {date}\n"
            f"[←/→] Navigate  [Drag] Select segment  [Shift+Drag] Add segment  "
            f"[R] Reset  [C] Calc selected  [Q] Quit",
            fontsize=11
        )
        self.fig.canvas.draw_idle()

    def _on_span(self, ax_idx, xmin, xmax):
        if xmax - xmin < 0.5:
            return  # ignore tiny accidental drags

        if not self.shift_held:
            # Clear previous selections on this axis
            for p in self.highlight_patches.get(ax_idx, []):
                p.remove()
            self.selections[ax_idx] = []
            self.highlight_patches[ax_idx] = []

        self.selections.setdefault(ax_idx, []).append((xmin, xmax))

        # Draw highlight rectangle
        ax = self.axes[ax_idx]
        ylim = ax.get_ylim()
        rect = Rectangle((xmin, ylim[0]), xmax - xmin, ylim[1] - ylim[0],
                          alpha=0.2, facecolor='gold', edgecolor='orange',
                          linewidth=1.5)
        ax.add_patch(rect)
        self.highlight_patches.setdefault(ax_idx, []).append(rect)

        # Auto-calculate for this axis
        self._calc_selected_rmssd(ax_idx)
        self.fig.canvas.draw_idle()

    def _calc_selected_rmssd(self, ax_idx):
        group = self.groups[self.current_idx]
        if ax_idx >= len(group['sessions']):
            return
        sess = group['sessions'][ax_idx]
        segs = self.selections.get(ax_idx, [])
        if not segs:
            return

        ts = sess['ts']
        peaks = sess['peaks']
        t_rel = ts - ts[0]

        # Collect peaks within any selected segment
        selected_peaks = []
        for (xmin, xmax) in segs:
            mask = (t_rel[peaks] >= xmin) & (t_rel[peaks] <= xmax)
            selected_peaks.extend(peaks[mask].tolist())
        selected_peaks = sorted(set(selected_peaks))

        if len(selected_peaks) < 3:
            seg_rmssd = None
            seg_hr = None
            seg_mean_rr = None
        else:
            peak_arr = np.array(selected_peaks)
            seg_rmssd = compute_rmssd(ts, peak_arr)
            seg_hr = compute_hr(ts, peak_arr)
            seg_mean_rr = compute_mean_rr(ts, peak_arr)

        rmssd_s = f"{seg_rmssd:.2f}" if seg_rmssd is not None else "N/A"
        hr_s = f"{seg_hr:.1f}" if seg_hr is not None else "N/A"
        rr_s = f"{seg_mean_rr:.1f}" if seg_mean_rr is not None else "N/A"
        n_segs = len(segs)
        n_peaks = len(selected_peaks)
        total_dur = sum(xmax - xmin for xmin, xmax in segs)

        phase = sess['phase'].upper()
        rec = sess['rec']
        orig_rmssd = sess['rmssd']
        orig_s = f"{orig_rmssd:.2f}" if orig_rmssd is not None else "N/A"
        sbp = sess.get('sbp')
        sbp_s = f"{sbp:.0f}" if sbp is not None else "N/A"

        # Store selected mean_rr for cvBRS calculation
        sess['selected_mean_rr'] = seg_mean_rr

        phase_color = '#2e7d32' if sess['phase'] == 'before' else '#c62828'

        self.axes[ax_idx].set_title(
            f"[{phase}]  {rec['patient_folder']}  |  "
            f"Full RMSSD={orig_s}  |  "
            f"Sel: {n_segs}seg {total_dur:.1f}s {n_peaks}pk → "
            f"RMSSD={rmssd_s} meanRR={rr_s} HR={hr_s} SBP={sbp_s}",
            fontsize=9, fontweight='bold', color=phase_color
        )

        print(f"  [{phase}] {rec['patient_folder']}: "
              f"sel {n_segs} seg ({total_dur:.1f}s, {n_peaks} peaks) → "
              f"RMSSD={rmssd_s} ms, meanRR={rr_s} ms, HR={hr_s} bpm  "
              f"(full: {orig_s} ms)")

        # If both before and after have selected mean_rr, compute cvBRS
        self._try_print_cvbrs()

    def _try_print_cvbrs(self):
        """If both before and first-after sessions have mean_rr (selected or full)
        and SBP, compute and print cvBRS = ΔRR / ΔSBP."""
        group = self.groups[self.current_idx]
        sessions = group['sessions']
        if len(sessions) < 2:
            return

        before = sessions[0]
        after = sessions[1]

        # Use selected mean_rr if available, else full mean_rr
        rr_b = before.get('selected_mean_rr') or before.get('mean_rr')
        rr_a = after.get('selected_mean_rr') or after.get('mean_rr')
        sbp_b = before.get('sbp')
        sbp_a = after.get('sbp')

        if rr_b is None or rr_a is None:
            return
        if sbp_b is None or sbp_a is None:
            print(f"  cvBRS: N/A (SBP missing: before={sbp_b}, after={sbp_a})")
            return

        delta_rr = rr_a - rr_b
        delta_sbp = sbp_a - sbp_b
        if abs(delta_sbp) < 0.01:
            print(f"  cvBRS: N/A (ΔSBP ≈ 0)  |  ΔRR={delta_rr:+.2f} ms  ΔSBP={delta_sbp:+.1f} mmHg")
        else:
            cvbrs = delta_rr / delta_sbp
            print(f"  cvBRS = {cvbrs:.2f} ms/mmHg  |  ΔRR={delta_rr:+.2f} ms  ΔSBP={delta_sbp:+.1f} mmHg")

    def _on_key(self, event):
        if event.key == 'shift':
            self.shift_held = True
        elif event.key == 'right':
            if self.current_idx < len(self.groups) - 1:
                self.current_idx += 1
                # Restore visibility of all axes
                for ax in self.axes:
                    ax.set_visible(True)
                self._draw_group()
        elif event.key == 'left':
            if self.current_idx > 0:
                self.current_idx -= 1
                for ax in self.axes:
                    ax.set_visible(True)
                self._draw_group()
        elif event.key == 'r':
            self._clear_selections()
            self._draw_group()
            print("  [Reset] Cleared all selections.")
        elif event.key == 'c':
            print("\n--- Recalculate from selected segments ---")
            group = self.groups[self.current_idx]
            for i in range(len(group['sessions'])):
                self._calc_selected_rmssd(i)
            print("---\n")
        elif event.key == 'q':
            plt.close(self.fig)

    def _on_key_release(self, event):
        if event.key == 'shift':
            self.shift_held = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate HRV RMSSD for stimulated patients in mirror4."
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Launch interactive visualization for each group."
    )
    args = parser.parse_args()

    # Phase 1: scan mirror4 patient folders
    records = []
    patient_re = re.compile(r"^patient_\d+$")

    for folder in sorted(os.listdir(MIRROR4_DIR)):
        if not patient_re.match(folder):
            continue
        patient_dir = os.path.join(MIRROR4_DIR, folder)
        if not os.path.isdir(patient_dir):
            continue

        patient_id, hosp_id, session_ts = read_patient_info(patient_dir)
        if not patient_id or not hosp_id or not session_ts:
            continue

        try:
            hosp_id_int = int(hosp_id)
        except ValueError:
            continue

        try:
            session_dt = datetime.strptime(session_ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        records.append({
            "patient_id": patient_id,
            "hospital_patient_id": hosp_id,
            "hospital_patient_id_int": hosp_id_int,
            "session_datetime": session_dt,
            "session_date": session_dt.strftime("%Y-%m-%d"),
            "session_time": session_dt.strftime("%H:%M:%S"),
            "patient_folder": folder,
            "patient_dir": patient_dir,
        })

    # Phase 2: group by (hospital_patient_id_int, date)
    groups = defaultdict(list)
    for rec in records:
        key = (rec["hospital_patient_id_int"], rec["session_date"])
        groups[key].append(rec)

    # Filter to groups with 2+ sessions (stimulated)
    stim_groups = {k: sorted(v, key=lambda r: r["session_datetime"])
                   for k, v in groups.items() if len(v) >= 2}

    print(f"Found {len(stim_groups)} stimulated groups in mirror4.")

    # Load SBP data from merged patient info
    sbp_map = load_sbp_map(MERGED_CSV)
    print(f"Loaded SBP data for {sum(1 for v in sbp_map.values() if v is not None)} patients.")

    # Phase 3: compute RMSSD for each session (and cache signal data for vis)
    results = []
    vis_groups_data = []  # for visualization

    for (hosp_id_int, date), recs in sorted(stim_groups.items()):
        n_sessions = len(recs)
        vis_sessions = []

        for idx, rec in enumerate(recs):
            # First session = before, rest = after
            if idx == 0:
                phase = "before"
            else:
                phase = f"after_{idx}" if n_sessions > 2 else "after"

            ts, ecg = read_ecg(rec["patient_dir"])
            rmssd = None
            hr = None
            mean_rr = None
            n_peaks = 0
            duration_s = 0.0
            peaks = np.array([])
            pantompkins = np.array([])

            if ts is not None and ecg is not None:
                duration_s = ts[-1] - ts[0]
                peaks, pantompkins = detect_r_peaks(ecg)
                n_peaks = len(peaks)
                rmssd = compute_rmssd(ts, peaks)
                hr = compute_hr(ts, peaks)
                mean_rr = compute_mean_rr(ts, peaks)

            # Look up SBP for this patient
            try:
                pid_int = int(rec["patient_id"])
            except ValueError:
                pid_int = -1
            sbp = sbp_map.get(pid_int, None)

            rmssd_str = f"{rmssd:.2f}" if rmssd is not None else "N/A"
            hr_str = f"{hr:.1f}" if hr is not None else "N/A"
            mean_rr_str = f"{mean_rr:.2f}" if mean_rr is not None else "N/A"
            sbp_str = f"{sbp:.0f}" if sbp is not None else "N/A"

            results.append({
                "hospital_patient_id": rec["hospital_patient_id"],
                "patient_id": rec["patient_id"],
                "session_date": rec["session_date"],
                "session_time": rec["session_time"],
                "phase": phase,
                "rmssd_ms": rmssd_str,
                "mean_rr_ms": mean_rr_str,
                "hr_bpm": hr_str,
                "sbp_mmhg": sbp_str,
                "n_peaks": n_peaks,
                "duration_s": f"{duration_s:.1f}",
                "patient_folder": rec["patient_folder"],
            })

            vis_sessions.append({
                'rec': rec,
                'ts': ts if ts is not None else np.array([0, 1]),
                'ecg': ecg if ecg is not None else np.array([0, 0]),
                'pantompkins': pantompkins if len(pantompkins) > 0 else np.array([]),
                'peaks': peaks,
                'phase': phase,
                'rmssd': rmssd,
                'mean_rr': mean_rr,
                'hr': hr,
                'sbp': sbp,
            })

            print(f"  {rec['patient_folder']} | hosp_id={rec['hospital_patient_id']} | "
                  f"{date} {rec['session_time']} | {phase} | "
                  f"RMSSD={rmssd_str} ms | meanRR={mean_rr_str} ms | "
                  f"HR={hr_str} bpm | SBP={sbp_str} mmHg | peaks={n_peaks}")

        vis_groups_data.append({
            'key': (hosp_id_int, date),
            'sessions': vis_sessions,
        })

    # Phase 4: write CSV
    fieldnames = [
        "hospital_patient_id",
        "patient_id",
        "session_date",
        "session_time",
        "phase",
        "rmssd_ms",
        "mean_rr_ms",
        "hr_bpm",
        "sbp_mmhg",
        "n_peaks",
        "duration_s",
        "patient_folder",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Phase 5: summary — show before vs after comparison with cvBRS
    print("\n" + "=" * 90)
    print("SUMMARY: Before vs After Stimulation (RMSSD, mean RR, cvBRS)")
    print("=" * 90)

    for (hosp_id_int, date), recs in sorted(stim_groups.items()):
        hosp_id = recs[0]["hospital_patient_id"]
        group_results = [r for r in results
                         if r["hospital_patient_id"] == hosp_id
                         and r["session_date"] == date]
        before = [r for r in group_results if r["phase"] == "before"]
        after = [r for r in group_results if r["phase"].startswith("after")]

        before_rmssd = before[0]["rmssd_ms"] if before else "N/A"
        before_rr = before[0]["mean_rr_ms"] if before else "N/A"
        before_sbp = before[0]["sbp_mmhg"] if before else "N/A"
        after_rmssds = [r["rmssd_ms"] for r in after]
        after_rrs = [r["mean_rr_ms"] for r in after]
        after_sbps = [r["sbp_mmhg"] for r in after]

        print(f"\nHospital ID: {hosp_id} | Date: {date}")
        print(f"  Before:  RMSSD={before_rmssd} ms | meanRR={before_rr} ms | SBP={before_sbp} mmHg")
        for i in range(len(after_rmssds)):
            label = f"After {i+1}" if len(after_rmssds) > 1 else "After"
            print(f"  {label}:  RMSSD={after_rmssds[i]} ms | meanRR={after_rrs[i]} ms | SBP={after_sbps[i]} mmHg")

        # Compute RMSSD change
        try:
            b = float(before_rmssd)
            a = float(after_rmssds[0])
            change = a - b
            pct = (change / b) * 100 if b != 0 else float("inf")
            print(f"  ΔRMSSD:  {change:+.2f} ms ({pct:+.1f}%)")
        except (ValueError, IndexError):
            print(f"  ΔRMSSD:  N/A")

        # Compute cvBRS = ΔRR(ms) / ΔSBP(mmHg)
        try:
            rr_b = float(before_rr)
            rr_a = float(after_rrs[0])
            sbp_b = float(before_sbp)
            sbp_a = float(after_sbps[0])
            delta_rr = rr_a - rr_b
            delta_sbp = sbp_a - sbp_b
            if abs(delta_sbp) < 0.01:
                print(f"  cvBRS:   N/A (ΔSBP ≈ 0)  |  ΔRR={delta_rr:+.2f} ms  ΔSBP={delta_sbp:+.1f} mmHg")
            else:
                cvbrs = delta_rr / delta_sbp
                print(f"  cvBRS:   {cvbrs:.2f} ms/mmHg  |  ΔRR={delta_rr:+.2f} ms  ΔSBP={delta_sbp:+.1f} mmHg")
        except (ValueError, IndexError):
            print(f"  cvBRS:   N/A (missing RR or SBP data)")

    print(f"\nResults saved to: {OUTPUT_CSV}")

    # Phase 6: interactive visualization
    if args.vis:
        print("\nLaunching interactive visualization...")
        print("  [←/→] Navigate groups  |  [Drag] Select segment  |  "
              "[Shift+Drag] Add segment")
        print("  [R] Reset selections   |  [C] Recalculate         |  [Q] Quit")
        StimHRVViewer(vis_groups_data)


if __name__ == "__main__":
    main()
