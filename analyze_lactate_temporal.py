"""
Temporal Lactate Analysis — Δ-Lactate vs Δ-Vitals.

Parses raw XLSX (individual measurements with timestamps) and session timestamps
from patient_info.txt files, temporally matches lactate to recording sessions,
then computes delta features for patients with multiple sessions.

Usage:
    python analyze_lactate_temporal.py
    # Or with environment:
    /home/sqcheng/miniforge3/envs/healthmirror-env/bin/python analyze_lactate_temporal.py
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'lactate_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIRROR_IDS = [1, 2, 4, 5, 6]
XLSX_PATH = os.path.join(BASE_DIR, '健康镜化验1.0.xlsx')

# ── Step 1: Parse raw XLSX for individual measurements ──────────────────

def parse_lactate_measurements(xlsx_path):
    """
    Parse raw XLSX and return DataFrame of individual lactate measurements.
    
    Returns:
        pd.DataFrame with columns:
            hospital_patient_id (int)
            lactate_value (float)
            report_time (datetime)
            gender (str)
            age (float)
    """
    df = pd.read_excel(xlsx_path, header=[0, 1])
    df.columns = [col[1] if col[1] else col[0] for col in df.columns]

    # Convert patient ID
    df['hospital_patient_id'] = pd.to_numeric(df['首页病案号'], errors='coerce')
    df = df.dropna(subset=['hospital_patient_id'])
    df['hospital_patient_id'] = df['hospital_patient_id'].astype(int)

    # Parse lactate value
    df['lactate_value'] = pd.to_numeric(df['检验值(文本)'], errors='coerce')

    # Parse report time
    df['report_time'] = pd.to_datetime(df['报告时间'], errors='coerce')

    # Drop invalid measurements
    df = df.dropna(subset=['lactate_value', 'report_time'])
    df = df[df['lactate_value'] > 0]  # positive lactate only

    # Extract age
    df['age'] = df['首页就诊时年龄'].astype(str).str.replace('岁', '', regex=False)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['gender'] = df['首页性别']

    result = df[['hospital_patient_id', 'lactate_value', 'report_time', 'gender', 'age']].copy()
    result = result.sort_values(['hospital_patient_id', 'report_time']).reset_index(drop=True)

    print(f"  [XLSX] Parsed {len(result)} individual lactate measurements "
          f"from {result['hospital_patient_id'].nunique()} unique patients")
    print(f"  [XLSX] Lactate range: [{result['lactate_value'].min():.2f}, "
          f"{result['lactate_value'].max():.2f}] mmol/L")
    print(f"  [XLSX] Time range: {result['report_time'].min()} to {result['report_time'].max()}")

    return result


# ── Step 2: Extract session timestamps from mirror data directories ─────

def parse_session_timestamp(line):
    """Parse 'Session Timestamp: 2025-08-18 20:06:29' -> pd.Timestamp."""
    match = re.search(r'Session Timestamp:\s*(.+)', line)
    if match:
        try:
            return pd.to_datetime(match.group(1).strip())
        except Exception:
            return pd.NaT
    return pd.NaT


def parse_vitals_from_json(json_str):
    """
    Parse vitals from patient JSON string.
    Returns dict with keys: spo2, hr, rr, temp, sbp, dbp
    Missing values default to NaN.
    """
    result = {'spo2': np.nan, 'hr': np.nan, 'rr': np.nan,
              'temp': np.nan, 'sbp': np.nan, 'dbp': np.nan}

    try:
        data = json.loads(json_str)
        vitals = data.get('vitals', {})

        spo2_str = vitals.get('blood_oxygen', 'n/a')
        if spo2_str != 'n/a':
            result['spo2'] = int(spo2_str.strip('%'))

        hr_str = vitals.get('heart_rate', 'n/a')
        if hr_str != 'n/a':
            result['hr'] = int(hr_str.strip('bpm'))

        rr_str = vitals.get('respiratory_rate', 'n/a')
        if rr_str != 'n/a':
            result['rr'] = int(rr_str.strip('bpm'))

        temp_str = vitals.get('temperature', 'n/a')
        if temp_str != 'n/a':
            result['temp'] = float(temp_str.replace('℃', '').replace('°C', ''))

        bp_str = vitals.get('blood_pressure', 'n/a')
        if bp_str != 'n/a' and '/' in bp_str:
            parts = bp_str.split('/')
            result['sbp'] = int(parts[0])
            result['dbp'] = int(parts[1])

    except Exception:
        pass

    return result


def extract_all_sessions():
    """
    Walk all mirror data directories and extract session timestamps + vitals.
    
    Returns:
        pd.DataFrame with columns:
            lab_patient_id (str)
            hospital_patient_id (int/str)
            mirror_id (int)
            session_timestamp (datetime)
            sbp, dbp, hr, spo2, rr, temp (float)
    """
    sessions = []

    for mirror_id in MIRROR_IDS:
        data_dir = os.path.join(BASE_DIR, f'mirror{mirror_id}_data')
        if not os.path.isdir(data_dir):
            print(f"  [WARN] Mirror {mirror_id} data dir not found: {data_dir}")
            continue

        patient_dirs = sorted([
            d for d in os.listdir(data_dir)
            if d.startswith('patient_') and os.path.isdir(os.path.join(data_dir, d))
        ])

        for pdir in patient_dirs:
            info_path = os.path.join(data_dir, pdir, 'patient_info.txt')
            if not os.path.isfile(info_path):
                continue

            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                lab_patient_id = pdir.replace('patient_', '')
                session_ts = pd.NaT
                hospital_patient_id = None
                vitals = {}

                for line in lines:
                    if line.startswith('Patient ID:'):
                        pid_from_file = line.split(':', 1)[1].strip()
                        lab_patient_id = pid_from_file
                    elif line.startswith('Session Timestamp:'):
                        session_ts = parse_session_timestamp(line)
                    elif line.startswith('Patient Info:'):
                        json_str = (line.split(':', 1)[1].strip()
                                    .strip('"')
                                    .replace('\\"', '"')
                                    .replace('\\/', '/'))
                        try:
                            data = json.loads(json_str)
                            hospital_patient_id = data.get('patient_id', None)
                            vitals = parse_vitals_from_json(json_str)
                        except Exception:
                            pass

                if hospital_patient_id is not None and pd.notna(session_ts):
                    sessions.append({
                        'lab_patient_id': str(lab_patient_id),
                        'hospital_patient_id': str(hospital_patient_id),
                        'mirror_id': mirror_id,
                        'session_timestamp': session_ts,
                        **vitals
                    })
            except Exception as e:
                print(f"  [WARN] Error reading {info_path}: {e}")
                continue

    result = pd.DataFrame(sessions)
    # Convert hospital_patient_id to int where possible
    result['hospital_patient_id'] = pd.to_numeric(result['hospital_patient_id'], errors='coerce')
    result = result.dropna(subset=['hospital_patient_id'])
    result['hospital_patient_id'] = result['hospital_patient_id'].astype(int)

    print(f"  [SESSIONS] Extracted {len(result)} sessions from {result['hospital_patient_id'].nunique()} unique patients")
    print(f"  [SESSIONS] Per mirror: {result.groupby('mirror_id').size().to_dict()}")
    print(f"  [SESSIONS] Time range: {result['session_timestamp'].min()} to {result['session_timestamp'].max()}")

    return result


# ── Step 3: Temporally match lactate to recording sessions ──────────────

def match_lactate_to_sessions(measurements, sessions):
    """
    For each (patient, session), find the closest lactate measurement.
    
    Match rules:
        Primary: lactate measured within 24h before the session
        Secondary: nearest lactate measurement within 7 days (before or after)
        Tertiary: closest measurement regardless of time window
    
    Returns:
        sessions with matched lactate info appended (one row per session)
    """
    matched_rows = []

    # Group measurements by patient for efficiency
    meas_by_patient = measurements.groupby('hospital_patient_id')

    total_sessions = len(sessions)
    matched_count = 0

    for idx, session in sessions.iterrows():
        pid = session['hospital_patient_id']
        session_ts = session['session_timestamp']

        if pid not in meas_by_patient.groups:
            continue

        patient_meas = meas_by_patient.get_group(pid).copy()
        time_diffs = (patient_meas['report_time'] - session_ts).dt.total_seconds() / 3600  # hours
        abs_time_diffs = time_diffs.abs()

        best_match_row = None
        best_match_type = None
        best_match_time_diff = None

        # Primary: within 24h before session
        primary_mask = (time_diffs >= -24) & (time_diffs <= 0)
        if primary_mask.any():
            # Closest before session: smallest absolute time_diff among primary candidates
            primary_abs = abs_time_diffs[primary_mask]
            best_pos = primary_abs.idxmin()  # returns label index
            best_match_row = patient_meas.loc[best_pos]
            best_match_type = 'primary'
            best_match_time_diff = time_diffs[best_pos]
        else:
            # Secondary: nearest within 7 days (either direction)
            secondary_mask = abs_time_diffs <= 24 * 7
            if secondary_mask.any():
                best_pos = abs_time_diffs[secondary_mask].idxmin()
                best_match_row = patient_meas.loc[best_pos]
                best_match_time_diff = time_diffs[best_pos]
                best_match_type = 'secondary'
            else:
                # Tertiary: closest regardless of time
                best_pos = abs_time_diffs.idxmin()
                best_match_row = patient_meas.loc[best_pos]
                best_match_time_diff = time_diffs[best_pos]
                best_match_type = 'tertiary'

        if best_match_row is not None:
            matched_rows.append({
                'lab_patient_id': session['lab_patient_id'],
                'hospital_patient_id': pid,
                'mirror_id': session['mirror_id'],
                'session_timestamp': session_ts,
                'sbp': session.get('sbp', np.nan),
                'dbp': session.get('dbp', np.nan),
                'hr': session.get('hr', np.nan),
                'spo2': session.get('spo2', np.nan),
                'rr': session.get('rr', np.nan),
                'temp': session.get('temp', np.nan),
                'lactate_value': best_match_row['lactate_value'],
                'lactate_report_time': best_match_row['report_time'],
                'match_type': best_match_type,
                'time_diff_hours': best_match_time_diff,
            })
            matched_count += 1

    result = pd.DataFrame(matched_rows)
    # Sort by patient, then session time
    result = result.sort_values(['hospital_patient_id', 'session_timestamp']).reset_index(drop=True)

    print(f"\n  [MATCH] Total sessions: {total_sessions}")
    print(f"  [MATCH] Matched: {matched_count} ({matched_count/total_sessions*100:.1f}%)")
    if len(result) > 0:
        print(f"  [MATCH] Match types: {result['match_type'].value_counts().to_dict()}")
        match_times = result['time_diff_hours'].abs()
        print(f"  [MATCH] |Δ-time| (hours): mean={match_times.mean():.1f}, "
              f"median={match_times.median():.1f}, "
              f"range=[{match_times.min():.1f}, {match_times.max():.1f}]")

    return result


# ── Step 4: Compute Δ-features for patients with multiple sessions ──────

def compute_delta_features(matched_sessions):
    """
    For each patient with ≥2 temporally-matched sessions,
    compute Δ-lactate and Δ-vitals between consecutive sessions.
    
    Returns:
        pd.DataFrame of delta features (one row per consecutive pair)
    """
    delta_rows = []

    for pid, group in matched_sessions.groupby('hospital_patient_id'):
        group = group.sort_values('session_timestamp').reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            s1 = group.iloc[i]
            s2 = group.iloc[i + 1]

            delta_days = (s2['session_timestamp'] - s1['session_timestamp']).total_seconds() / 86400

            delta_rows.append({
                'hospital_patient_id': pid,
                'mirror_id_1': s1['mirror_id'],
                'mirror_id_2': s2['mirror_id'],
                'session_time_1': s1['session_timestamp'],
                'session_time_2': s2['session_timestamp'],
                'delta_days': delta_days,
                'match_type_1': s1['match_type'],
                'match_type_2': s2['match_type'],
                'time_diff_hours_1': s1['time_diff_hours'],
                'time_diff_hours_2': s2['time_diff_hours'],
                'lactate_1': s1['lactate_value'],
                'lactate_2': s2['lactate_value'],
                'delta_lactate': s2['lactate_value'] - s1['lactate_value'],
                'sbp_1': s1['sbp'],
                'sbp_2': s2['sbp'],
                'delta_sbp': s2['sbp'] - s1['sbp'] if (pd.notna(s1['sbp']) and pd.notna(s2['sbp'])) else np.nan,
                'dbp_1': s1['dbp'],
                'dbp_2': s2['dbp'],
                'delta_dbp': s2['dbp'] - s1['dbp'] if (pd.notna(s1['dbp']) and pd.notna(s2['dbp'])) else np.nan,
                'hr_1': s1['hr'],
                'hr_2': s2['hr'],
                'delta_hr': s2['hr'] - s1['hr'] if (pd.notna(s1['hr']) and pd.notna(s2['hr'])) else np.nan,
                'spo2_1': s1['spo2'],
                'spo2_2': s2['spo2'],
                'delta_spo2': s2['spo2'] - s1['spo2'] if (pd.notna(s1['spo2']) and pd.notna(s2['spo2'])) else np.nan,
                'rr_1': s1['rr'],
                'rr_2': s2['rr'],
                'delta_rr': s2['rr'] - s1['rr'] if (pd.notna(s1['rr']) and pd.notna(s2['rr'])) else np.nan,
                'temp_1': s1['temp'],
                'temp_2': s2['temp'],
                'delta_temp': s2['temp'] - s1['temp'] if (pd.notna(s1['temp']) and pd.notna(s2['temp'])) else np.nan,
            })

    delta_df = pd.DataFrame(delta_rows)

    n_patients = delta_df['hospital_patient_id'].nunique()
    n_pairs = len(delta_df)

    print(f"\n  [DELTA] Patients with ≥2 sessions: {n_patients}")
    print(f"  [DELTA] Total consecutive session pairs: {n_pairs}")
    if n_pairs > 0:
        print(f"  [DELTA] Δ-lactate: mean={delta_df['delta_lactate'].mean():.3f} mmol/L, "
              f"std={delta_df['delta_lactate'].std():.3f}, "
              f"range=[{delta_df['delta_lactate'].min():.3f}, {delta_df['delta_lactate'].max():.3f}]")
        print(f"  [DELTA] Δ-time (days): mean={delta_df['delta_days'].mean():.1f}, "
              f"median={delta_df['delta_days'].median():.1f}")
        print(f"  [DELTA] Available Δ-vital pairs:")
        for vital in ['sbp', 'dbp', 'hr', 'spo2', 'rr', 'temp']:
            n_valid = delta_df[f'delta_{vital}'].notna().sum()
            print(f"    Δ-{vital}: {n_valid} valid pairs")

    return delta_df


# ── Step 5: Correlation & visualization ──────────────────────────────────

def analyze_delta_correlations(delta_df):
    """
    Compute Pearson/Spearman correlations of Δ-lactate vs each Δ-vital.
    Returns a results DataFrame.
    """
    delta_vitals = ['delta_sbp', 'delta_dbp', 'delta_hr', 'delta_spo2', 'delta_rr', 'delta_temp']
    vital_labels = ['Δ-SBP (mmHg)', 'Δ-DBP (mmHg)', 'Δ-HR (bpm)',
                    'Δ-SpO2 (%)', 'Δ-RR (bpm)', 'Δ-Temp (°C)']
    vital_keys = ['SBP', 'DBP', 'HR', 'SpO2', 'RR', 'Temp']

    results = []
    for col, label, key in zip(delta_vitals, vital_labels, vital_keys):
        valid = delta_df[['delta_lactate', col]].dropna()
        n = len(valid)

        if n < 5:
            results.append({
                'Δ-Vital': key,
                'N (pairs)': n,
                'Pearson r': np.nan,
                'Pearson p': np.nan,
                'Spearman ρ': np.nan,
                'Spearman p': np.nan,
                'Interpretation': 'Insufficient data'
            })
            continue

        r_pearson, p_pearson = pearsonr(valid['delta_lactate'], valid[col])
        r_spearman, p_spearman = spearmanr(valid['delta_lactate'], valid[col])

        # Interpretation
        abs_r = abs(r_pearson)
        if p_pearson > 0.05:
            interp = 'Not significant'
        elif abs_r < 0.1:
            interp = 'Very weak'
        elif abs_r < 0.3:
            interp = 'Weak'
        elif abs_r < 0.5:
            interp = 'Moderate'
        else:
            interp = 'Strong'

        results.append({
            'Δ-Vital': key,
            'N (pairs)': n,
            'Pearson r': round(r_pearson, 4),
            'Pearson p': round(p_pearson, 6),
            'Spearman ρ': round(r_spearman, 4),
            'Spearman p': round(p_spearman, 6),
            'Interpretation': interp
        })

    corr_df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"  Δ-LACTATE vs Δ-VITAL CORRELATIONS")
    print(f"{'='*70}")
    print(corr_df.to_string(index=False))

    return corr_df


def plot_delta_scatter(delta_df, corr_df):
    """Scatter plots: Δ-lactate vs each Δ-vital with regression line."""
    delta_vitals = ['delta_sbp', 'delta_dbp', 'delta_hr', 'delta_spo2', 'delta_rr', 'delta_temp']
    vital_labels = ['Δ-SBP (mmHg)', 'Δ-DBP (mmHg)', 'Δ-HR (bpm)',
                    'Δ-SpO2 (%)', 'Δ-RR (bpm)', 'Δ-Temp (°C)']
    filenames = ['delta_lactate_vs_delta_sbp.png', 'delta_lactate_vs_delta_dbp.png',
                 'delta_lactate_vs_delta_hr.png', 'delta_lactate_vs_delta_spo2.png',
                 'delta_lactate_vs_delta_rr.png', 'delta_lactate_vs_delta_temp.png']

    for col, label, fname in zip(delta_vitals, vital_labels, filenames):
        valid = delta_df[['delta_lactate', col]].dropna()
        n = len(valid)

        fig, ax = plt.subplots(figsize=(8, 6))

        x = valid['delta_lactate'].values
        y = valid[col].values

        ax.scatter(x, y, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.5)

        if n >= 5:
            # Linear regression
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x)
            ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)

            # Find correlation
            row = corr_df[corr_df['Δ-Vital'] == col.replace('delta_', '').upper()]
            if len(row) > 0:
                r_val = row['Pearson r'].values[0]
                p_val = row['Pearson p'].values[0]
                if not np.isnan(r_val):
                    title_str = f'{label} vs Δ-Lactate\n'
                    if p_val <= 0.001:
                        title_str += f'Pearson r={r_val:.3f}, p<0.001'
                    elif p_val <= 0.05:
                        title_str += f'Pearson r={r_val:.3f}, p={p_val:.4f}'
                    else:
                        title_str += f'Pearson r={r_val:.3f}, p={p_val:.4f} (ns)'
                    ax.set_title(title_str, fontsize=12)

        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Δ-Lactate (mmol/L)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add annotation about n
        ax.text(0.02, 0.98, f'n={n}', transform=ax.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        path = os.path.join(OUTPUT_DIR, fname)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  [PLOT] Saved: {path}")


def plot_delta_correlation_heatmap(delta_df):
    """Heatmap of all delta-feature correlations."""
    delta_cols = ['delta_lactate', 'delta_sbp', 'delta_dbp', 'delta_hr',
                  'delta_spo2', 'delta_rr', 'delta_temp']
    valid = delta_df[delta_cols].dropna()
    if len(valid) < 5:
        print("  [WARN] Too few samples for delta correlation heatmap")
        return

    corr = valid.corr(method='pearson')

    labels = ['Δ-Lactate', 'Δ-SBP', 'Δ-DBP', 'Δ-HR', 'Δ-SpO2', 'Δ-RR', 'Δ-Temp']

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=text_color, fontsize=9)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title('Δ-Feature Pearson Correlation Matrix', fontsize=13)

    path = os.path.join(OUTPUT_DIR, 'delta_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved: {path}")


def plot_delta_subgroup_analysis(delta_df):
    """
    Categorize Δ-lactate into tertiles (small/medium/large change)
    and compare mean Δ-vitals across groups.
    """
    valid = delta_df.dropna(subset=['delta_lactate']).copy()
    if len(valid) < 10:
        print("  [WARN] Too few samples for subgroup analysis")
        return

    # Tertile split — handle duplicate bin edges (many delta_lactate==0)
    q33, q66 = valid['delta_lactate'].quantile([1/3, 2/3]).values
    # Check for duplicate boundaries when many samples are 0
    if q33 >= q66 or abs(q33 - q66) < 1e-6:
        # Fall back to sign-based split: negative / zero / positive
        labels = ['Decrease', 'No Change', 'Increase']
        valid['lactate_change_group'] = pd.cut(
            valid['delta_lactate'],
            bins=[-np.inf, -1e-6, 1e-6, np.inf],
            labels=labels,
        )
    else:
        labels = ['Decrease', 'Stable', 'Increase']
        valid['lactate_change_group'] = pd.cut(
            valid['delta_lactate'],
            bins=[-np.inf, q33, q66, np.inf],
            labels=labels,
        )

    delta_vitals = ['delta_sbp', 'delta_dbp', 'delta_hr', 'delta_spo2', 'delta_rr', 'delta_temp']
    vital_labels = ['Δ-SBP (mmHg)', 'Δ-DBP (mmHg)', 'Δ-HR (bpm)',
                    'Δ-SpO2 (%)', 'Δ-RR (bpm)', 'Δ-Temp (°C)']

    n_groups = len(labels)
    n_vitals = len(delta_vitals)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for ax, col, label in zip(axes, delta_vitals, vital_labels):
        data_by_group = []
        for g in labels:
            g_data = valid[valid['lactate_change_group'] == g][col].dropna().values
            data_by_group.append(g_data)

        # Bar plot with individual points
        x_pos = np.arange(n_groups)
        means = [np.mean(d) if len(d) > 0 else 0 for d in data_by_group]
        sems = [np.std(d) / np.sqrt(len(d)) if len(d) > 1 else 0 for d in data_by_group]

        bars = ax.bar(x_pos, means, yerr=sems, color=colors, alpha=0.7,
                      capsize=5, edgecolor='black', linewidth=1)

        # Individual points
        for i, d in enumerate(data_by_group):
            jitter = np.random.normal(i, 0.08, size=len(d))
            ax.scatter(jitter, d, alpha=0.3, s=15, color='black', zorder=5)

        # Anova-style note
        ns = [len(d) for d in data_by_group]
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labels, ns)], fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        # One-way ANOVA p-value
        from scipy.stats import f_oneway
        nonempty = [d for d in data_by_group if len(d) > 1]
        if len(nonempty) >= 2:
            try:
                f_stat, p_val = f_oneway(*nonempty)
                p_text = f'p={p_val:.4f}' if p_val >= 0.001 else 'p<0.001'
                ax.set_title(f'{label}\nANOVA {p_text}', fontsize=10)
            except Exception:
                ax.set_title(label, fontsize=10)
        else:
            ax.set_title(label, fontsize=10)

    plt.suptitle('Δ-Vitals by Δ-Lactate Change Group (Tertile Split)', fontsize=14, y=1.02)

    # Add lactate range annotation
    range_text = (f'Δ-Lactate ranges: Decrease (<0), No Change (~0), Increase (>0)')
    fig.text(0.5, -0.02, range_text, ha='center', fontsize=9, style='italic')

    path = os.path.join(OUTPUT_DIR, 'delta_subgroup_analysis.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [PLOT] Saved: {path}")


def plot_match_type_summary(matched_sessions):
    """Summary of temporal matching quality."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Match type distribution
    ax = axes[0]
    match_counts = matched_sessions['match_type'].value_counts()
    colors = {'primary': '#27ae60', 'secondary': '#2980b9', 'tertiary': '#e74c3c'}
    bar_colors = [colors.get(m, '#95a5a6') for m in match_counts.index]
    bars = ax.bar(range(len(match_counts)), match_counts.values, color=bar_colors,
                  edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(match_counts)))
    ax.set_xticklabels([f'{m}\n(n={match_counts[m]})' for m in match_counts.index])
    ax.set_ylabel('Number of Sessions')
    ax.set_title('Temporal Match Quality')
    ax.grid(True, alpha=0.3, axis='y')

    # Time difference distribution
    ax = axes[1]
    abs_times = np.abs(matched_sessions['time_diff_hours'].values)
    ax.hist(abs_times, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(24, color='red', linestyle='--', label='24h boundary')
    ax.axvline(24 * 7, color='orange', linestyle='--', label='7d boundary')
    ax.set_xlabel('|Time Difference| (hours)')
    ax.set_ylabel('Count')
    ax.set_title('Lactate-Session Time Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'match_type_summary.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] Saved: {path}")


# ── Step 6: Update note.md ──────────────────────────────────────────────

def generate_report_text(sessions, matched, delta_df, corr_df):
    """Generate the note.md appendix text."""
    lines = []
    lines.append("")
    lines.append("## 2026-05-12")
    lines.append("### Temporal Lactate Analysis (Δ-Lactate)")
    lines.append("- **Objective**: Use temporal alignment of individual lactate measurements with recording sessions to detect Δ-lactate ↔ Δ-vital relationships.")
    lines.append("")
    lines.append("#### Data & Matching")
    lines.append(f"- Lactate measurements parsed: {len(measurements)} from {measurements['hospital_patient_id'].nunique()} unique patients in XLSX")
    lines.append(f"- Recording sessions extracted: {len(sessions)} from {sessions['hospital_patient_id'].nunique()} patients (mirrors 1,2,4,5,6)")
    lines.append(f"- Temporally matched sessions: {len(matched)} ({len(matched)/len(sessions)*100:.1f}%)")

    match_type_counts = matched['match_type'].value_counts().to_dict()
    lines.append(f"  - Primary match (≤24h before): {match_type_counts.get('primary', 0)}")
    lines.append(f"  - Secondary match (≤7d): {match_type_counts.get('secondary', 0)}")
    lines.append(f"  - Tertiary match (beyond 7d): {match_type_counts.get('tertiary', 0)}")

    lines.append("")
    lines.append(f"#### Δ-Feature Analysis")
    lines.append(f"- Patients with ≥2 matched sessions: {delta_df['hospital_patient_id'].nunique()}")
    lines.append(f"- Consecutive session pairs: {len(delta_df)}")
    mean_delta_lac = delta_df['delta_lactate'].mean()
    lines.append(f"- Mean Δ-lactate: {mean_delta_lac:.3f} ± {delta_df['delta_lactate'].std():.3f} mmol/L")
    lines.append(f"- Δ-time between sessions: mean={delta_df['delta_days'].mean():.1f} days, median={delta_df['delta_days'].median():.1f} days")
    lines.append("")

    lines.append("#### Δ-Lactate vs Δ-Vital Correlations")
    lines.append("| Δ-Vital | N (pairs) | Pearson r | p-value | Spearman ρ | Interpretation |")
    lines.append("|---------|-----------|-----------|---------|------------|----------------|")
    for _, row in corr_df.iterrows():
        if pd.isna(row['Pearson r']):
            lines.append(f"| {row['Δ-Vital']} | {row['N (pairs)']} | — | — | — | {row['Interpretation']} |")
        else:
            p_str = f"p<0.001" if row['Pearson p'] < 0.001 else f"p={row['Pearson p']:.4f}"
            lines.append(f"| {row['Δ-Vital']} | {row['N (pairs)']} | {row['Pearson r']:.3f} | {p_str} | {row['Spearman ρ']:.3f} | {row['Interpretation']} |")

    lines.append("")
    lines.append("#### Key Findings")
    # Find strongest correlation
    valid_corr = corr_df.dropna(subset=['Pearson r'])
    if len(valid_corr) > 0:
        strongest = valid_corr.loc[valid_corr['Pearson r'].abs().idxmax()]
        p_text = f"p<0.001" if strongest['Pearson p'] < 0.001 else f"p={strongest['Pearson p']:.4f}"
        lines.append(f"- Strongest signal: {strongest['Δ-Vital']} (r={strongest['Pearson r']:.3f}, {p_text}) — {strongest['Interpretation']}")

    sig_count = len(valid_corr[valid_corr['Pearson p'] < 0.05])
    lines.append(f"- Significant correlations (p<0.05): {sig_count}/{len(valid_corr)}")
    lines.append("")

    # Compare with static analysis
    lines.append("#### Comparison with Static Analysis (2026-05-11)")
    lines.append("- Static analysis found very weak correlations (best: lactate_mean vs HR r=0.119, lactate_mean vs DBP r=0.098)")
    lines.append("- The temporal Δ-approach examines **within-patient changes over time**, removing inter-subject variability")
    if sig_count == 0:
        lines.append("- **Even with temporal alignment, no significant Δ-lactate ↔ Δ-vital relationships were detected** — "
                     "suggesting that lactate changes are not coupled with acute vital sign changes in this population")
        lines.append("- Possible reasons: (1) The patient population may have normal-range lactate with limited variability. "
                     "(2) Vital sign measurements from the mirror and lab lactate may not reflect the same physiological state. "
                     "(3) Sample size of paired sessions may be insufficient for robust delta analysis.")
    else:
        lines.append("- Temporal analysis reveals relationships not visible in static aggregate analysis, "
                     "confirming the value of temporal alignment.")

    lines.append("")
    lines.append("#### Output Files")
    lines.append("- `lactate_analysis/delta_lactate_vs_delta_*` — scatter plots for each vital")
    lines.append("- `lactate_analysis/delta_correlation_heatmap.png` — full Δ-feature correlation matrix")
    lines.append("- `lactate_analysis/delta_subgroup_analysis.png` — Δ-lactate tertile comparison")
    lines.append("- `lactate_analysis/match_type_summary.png` — quality of temporal matching")
    lines.append("")

    return '\n'.join(lines)


def update_note_md(report_text):
    """Append temporal analysis report to note.md."""
    note_path = os.path.join(BASE_DIR, 'note.md')

    if not os.path.exists(note_path):
        print(f"  [WARN] note.md not found at {note_path}")
        return

    with open(note_path, 'a', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  [NOTE] Appended results to {note_path}")


# ── Main pipeline ────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TEMPORAL LACTATE ANALYSIS (Δ-Lactate)")
    print("=" * 70)

    # Step 1: Parse raw XLSX
    print("\n[1] Parsing raw XLSX measurements...")
    global measurements
    measurements = parse_lactate_measurements(XLSX_PATH)

    # Step 2: Extract session timestamps
    print("\n[2] Extracting session timestamps from mirror data...")
    global sessions
    sessions = extract_all_sessions()

    # Step 3: Temporal matching
    print("\n[3] Temporally matching lactate to sessions...")
    global matched_sessions
    matched_sessions = match_lactate_to_sessions(measurements, sessions)

    if len(matched_sessions) < 5:
        print("\n  [WARN] Too few matched sessions for meaningful analysis.")
        report = generate_report_text_minimal(sessions, matched_sessions)
        update_note_md(report)
        return

    # Step 4: Compute delta features
    print("\n[4] Computing Δ-features...")
    delta_df = compute_delta_features(matched_sessions)

    if len(delta_df) < 5:
        print("\n  [WARN] Too few Δ-pairs for meaningful correlation analysis.")
        report = generate_report_text(sessions, matched_sessions, delta_df,
                                       pd.DataFrame(columns=['Δ-Vital', 'N (pairs)', 'Pearson r',
                                                              'Pearson p', 'Spearman ρ', 'Spearman p',
                                                              'Interpretation']))
        update_note_md(report)
        return

    # Step 5: Correlation & visualization
    print("\n[5] Correlation analysis & visualization...")

    # 5a: Correlation table
    print("\n  5a. Δ-lactate vs Δ-vital correlations...")
    corr_df = analyze_delta_correlations(delta_df)

    # 5b: Scatter plots
    print("\n  5b. Generating scatter plots...")
    plot_delta_scatter(delta_df, corr_df)

    # 5c: Heatmap
    print("\n  5c. Generating correlation heatmap...")
    plot_delta_correlation_heatmap(delta_df)

    # 5d: Subgroup analysis
    print("\n  5d. Generating subgroup analysis...")
    plot_delta_subgroup_analysis(delta_df)

    # 5e: Match summary
    print("\n  5e. Generating match summary plot...")
    plot_match_type_summary(matched_sessions)

    # Step 6: Update note.md
    print("\n[6] Updating note.md...")
    report = generate_report_text(sessions, matched_sessions, delta_df, corr_df)
    update_note_md(report)

    print(f"\n{'='*70}")
    print(f"  Analysis complete! Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


def generate_report_text_minimal(sessions, matched):
    """Fallback report when too few matches."""
    lines = []
    lines.append("")
    lines.append("## 2026-05-12")
    lines.append("### Temporal Lactate Analysis (Δ-Lactate)")
    lines.append("- **Status**: Insufficient temporally-matched data for Δ-analysis.")
    lines.append("")
    lines.append("#### Data & Matching")
    lines.append(f"- Lactate measurements parsed: {len(measurements)}")
    lines.append(f"- Recording sessions extracted: {len(sessions)}")
    lines.append(f"- Temporally matched sessions: {len(matched)}")
    lines.append("- Too few matched session pairs to compute meaningful Δ-correlations.")
    lines.append("- Conclusion: Temporal lactate alignment not feasible with current data.")
    lines.append("")
    return '\n'.join(lines)


if __name__ == '__main__':
    main()