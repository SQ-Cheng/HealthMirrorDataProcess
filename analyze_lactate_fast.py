"""
Fast Lactate Analysis - Compute statistics only (no plotting).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_all_lab_data():
    """Load and merge all mirror with_lab CSVs."""
    dfs = []
    for mirror_id in [1, 2, 4, 5, 6]:
        path = os.path.join(BASE_DIR, f'merged_patient_info_with_lab_{mirror_id}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['mirror_id'] = mirror_id
            dfs.append(df)
            print(f"  Mirror {mirror_id}: {df.shape[0]} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combined: {combined.shape[0]} rows")
    return combined


def main():
    print("=" * 60)
    print("  Lactate Analysis Pipeline (Fast Stats)")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_all_lab_data()
    
    # Filter valid lactate
    print("\n[2] Filtering valid lactate data...")
    valid = df[df['lactate_count'] > 0].copy()
    print(f"  Patients with lactate data: {valid.shape[0]}")
    print(f"  Unique hospital_patient_ids: {valid['hospital_patient_id'].nunique()}")
    
    # Basic stats
    print(f"\n{'='*60}")
    print("  Lactate Statistics (All Mirrors)")
    print(f"{'='*60}")
    
    for col in ['lactate_min', 'lactate_max', 'lactate_mean', 'lactate_median']:
        vals = valid[col].dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            print(f"  {col}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
                  f"std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")
    
    counts = valid['lactate_count'].dropna()
    counts = counts[counts > 0]
    if len(counts) > 0:
        print(f"  lactate_count: mean={counts.mean():.1f}, median={counts.median():.1f}, "
              f"range=[{counts.min():.0f}, {counts.max():.0f}]")
    
    # Detailed correlations
    valid_lac = valid[valid['lactate_mean'] > 0].copy()
    valid_bp = valid_lac[(valid_lac['low_blood_pressure'] > 0) & (valid_lac['high_blood_pressure'] > 0)]
    
    print(f"\n{'='*70}")
    print(f"  DETAILED CORRELATION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n  1. Data Overview:")
    print(f"     Total patients with lactate data: {len(valid_lac)}")
    print(f"     Patients with both lactate and BP: {len(valid_bp)}")
    print(f"     Lactate mean range: [{valid_lac['lactate_mean'].min():.2f}, {valid_lac['lactate_mean'].max():.2f}] mmol/L")
    print(f"     Lactate mean median: {valid_lac['lactate_mean'].median():.2f} mmol/L")
    
    print(f"\n  2. Lactate vs BP Correlations:")
    for lac_col in ['lactate_min', 'lactate_mean', 'lactate_max']:
        for bp_col in ['low_blood_pressure', 'high_blood_pressure']:
            mask = (valid_bp[lac_col].notna()) & (valid_bp[bp_col].notna())
            if mask.sum() > 5:
                r, p = pearsonr(valid_bp[lac_col][mask], valid_bp[bp_col][mask])
                rho, p_sp = spearmanr(valid_bp[lac_col][mask], valid_bp[bp_col][mask])
                bp_name = 'DBP' if 'low' in bp_col else 'SBP'
                print(f"     {lac_col} vs {bp_name}: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    print(f"\n  3. Lactate vs Age:")
    valid_age = valid_lac[valid_lac['age'] > 0]
    if len(valid_age) > 5:
        r, p = pearsonr(valid_age['lactate_mean'], valid_age['age'])
        rho, p_sp = spearmanr(valid_age['lactate_mean'], valid_age['age'])
        print(f"     Lactate mean vs Age: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    print(f"\n  4. Lactate by Gender:")
    valid_gender = valid_lac[valid_lac['gender'].isin(['男', '女'])]
    for gender in ['男', '女']:
        data = valid_gender[valid_gender['gender'] == gender]['lactate_mean'].dropna()
        gender_label = 'Male' if gender == '男' else 'Female'
        print(f"     {gender_label}: mean={data.mean():.2f}, median={data.median():.2f}, n={len(data)}")
    
    # Mann-Whitney U test for gender
    male_data = valid_gender[valid_gender['gender'] == '男']['lactate_mean'].dropna()
    female_data = valid_gender[valid_gender['gender'] == '女']['lactate_mean'].dropna()
    if len(male_data) > 5 and len(female_data) > 5:
        stat, p_val = mannwhitneyu(male_data, female_data)
        print(f"     Mann-Whitney U test (Male vs Female): p={p_val:.4f}")
    
    print(f"\n  5. Lactate vs Heart Rate:")
    valid_hr = valid_lac[valid_lac['heart_rate'] > 0]
    if len(valid_hr) > 5:
        r, p = pearsonr(valid_hr['lactate_mean'], valid_hr['heart_rate'])
        rho, p_sp = spearmanr(valid_hr['lactate_mean'], valid_hr['heart_rate'])
        print(f"     Lactate mean vs HR: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    print(f"\n  6. Lactate vs SpO2:")
    valid_spo2 = valid_lac[valid_lac['blood_oxygen'] > 0]
    if len(valid_spo2) > 5:
        r, p = pearsonr(valid_spo2['lactate_mean'], valid_spo2['blood_oxygen'])
        rho, p_sp = spearmanr(valid_spo2['lactate_mean'], valid_spo2['blood_oxygen'])
        print(f"     Lactate mean vs SpO2: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    print(f"\n  7. Lactate vs Respiratory Rate:")
    valid_resp = valid_lac[valid_lac['respiratory_rate'] > 0]
    if len(valid_resp) > 5:
        r, p = pearsonr(valid_resp['lactate_mean'], valid_resp['respiratory_rate'])
        rho, p_sp = spearmanr(valid_resp['lactate_mean'], valid_resp['respiratory_rate'])
        print(f"     Lactate mean vs RR: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    print(f"\n  8. Lactate vs Temperature:")
    valid_temp = valid_lac[valid_lac['temperature'] > 0]
    if len(valid_temp) > 5:
        r, p = pearsonr(valid_temp['lactate_mean'], valid_temp['temperature'])
        rho, p_sp = spearmanr(valid_temp['lactate_mean'], valid_temp['temperature'])
        print(f"     Lactate mean vs Temp: Pearson r={r:.3f}, p={p:.4f}, Spearman ρ={rho:.3f}")
    
    # Full correlation matrix
    print(f"\n  9. Full Correlation Matrix:")
    cols = ['age', 'low_blood_pressure', 'high_blood_pressure', 
            'heart_rate', 'blood_oxygen', 'respiratory_rate', 'temperature',
            'lactate_min', 'lactate_max', 'lactate_mean', 'lactate_median', 'lactate_count']
    
    corr_df = valid[cols].copy()
    corr_df = corr_df.replace(-1, np.nan)
    corr = corr_df.corr(method='pearson')
    
    labels = ['Age', 'DBP', 'SBP', 'HR', 'SpO2', 'RR', 'Temp',
              'Lac Min', 'Lac Max', 'Lac Mean', 'Lac Median', 'Lac Count']
    print(f"  {'':>12}", end='')
    for l in labels:
        print(f"{l:>10}", end='')
    print()
    for i, l in enumerate(labels):
        print(f"  {l:>12}", end='')
        for j in range(len(labels)):
            print(f"{corr.values[i, j]:>10.3f}", end='')
        print()
    
    # Top correlates for lactate
    print(f"\n  10. Top Correlates with Lactate:")
    lac_cols = ['lactate_min', 'lactate_mean', 'lactate_max', 'lactate_median']
    for lac in lac_cols:
        if lac in corr.columns:
            row = corr.loc[lac].drop(lac)
            top = row.abs().sort_values(ascending=False).head(3)
            print(f"     {lac}:")
            for idx, val in top.items():
                print(f"       {idx}: r={corr.loc[lac, idx]:.3f}")
    
    # Threshold analysis
    print(f"\n  11. Lactate Threshold Analysis (High vs Normal):")
    valid_thresh = valid_bp.copy()
    thresholds = [1.0, 1.5, 2.0, 2.2, 2.5, 3.0]
    for thresh in thresholds:
        high = valid_thresh[valid_thresh['lactate_mean'] >= thresh]
        normal = valid_thresh[valid_thresh['lactate_mean'] < thresh]
        if len(high) < 5 or len(normal) < 5:
            continue
        sbp_diff = high['high_blood_pressure'].mean() - normal['high_blood_pressure'].mean()
        dbp_diff = high['low_blood_pressure'].mean() - normal['low_blood_pressure'].mean()
        _, sbp_p = ttest_ind(high['high_blood_pressure'], normal['high_blood_pressure'])
        _, dbp_p = ttest_ind(high['low_blood_pressure'], normal['low_blood_pressure'])
        print(f"     Threshold >= {thresh:.1f} mmol/L:")
        print(f"       High group: n={len(high)}, Normal group: n={len(normal)}")
        print(f"       SBP diff: {sbp_diff:.2f} mmHg (p={sbp_p:.4f})")
        print(f"       DBP diff: {dbp_diff:.2f} mmHg (p={dbp_p:.4f})")
    
    print(f"\n{'='*60}")
    print(f"  Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
