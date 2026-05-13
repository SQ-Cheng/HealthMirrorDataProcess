"""
Analyze Lactate - Relationship between lactate values and other features.

Merges all mirror with_lab CSVs, then explores:
1. Lactate distribution & basic stats
2. Lactate vs BP (SBP/DBP)
3. Lactate vs Age
4. Lactate vs Gender
5. Lactate vs Heart Rate
6. Lactate vs Blood Oxygen
7. Multi-feature correlation matrix
8. Lactate threshold analysis (high vs normal lactate groups)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'lactate_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def filter_valid_lactate(df):
    """Filter to rows with valid lactate data (lactate_count > 0)."""
    valid = df[df['lactate_count'] > 0].copy()
    print(f"  Patients with lactate data: {valid.shape[0]}")
    print(f"  Unique hospital_patient_ids: {valid['hospital_patient_id'].nunique()}")
    return valid


def filter_valid_bp(df):
    """Filter to rows with valid BP data."""
    valid = df[(df['low_blood_pressure'] > 0) & (df['high_blood_pressure'] > 0)].copy()
    print(f"  Patients with valid BP: {valid.shape[0]}")
    return valid


def basic_stats(df, label):
    """Print basic statistics for lactate values."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    for col in ['lactate_min', 'lactate_max', 'lactate_mean', 'lactate_median']:
        vals = df[col].dropna()
        vals = vals[vals > 0]
        if len(vals) > 0:
            print(f"  {col}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
                  f"std={vals.std():.2f}, min={vals.min():.2f}, max={vals.max():.2f}")
    
    # Lactate count distribution
    counts = df['lactate_count'].dropna()
    counts = counts[counts > 0]
    if len(counts) > 0:
        print(f"  lactate_count: mean={counts.mean():.1f}, median={counts.median():.1f}, "
              f"range=[{counts.min():.0f}, {counts.max():.0f}]")


def plot_lactate_distribution(df, filename='lactate_distribution.png'):
    """Plot distribution of lactate metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = [
        ('lactate_min', 'Lactate Min (mmol/L)'),
        ('lactate_max', 'Lactate Max (mmol/L)'),
        ('lactate_mean', 'Lactate Mean (mmol/L)'),
        ('lactate_median', 'Lactate Median (mmol/L)'),
    ]
    
    for ax, (col, label) in zip(axes.flatten(), metrics):
        vals = df[col].dropna()
        vals = vals[vals > 0]
        ax.hist(vals, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(vals.median(), color='red', linestyle='--', label=f'Median={vals.median():.2f}')
        ax.axvline(vals.mean(), color='green', linestyle='--', label=f'Mean={vals.mean():.2f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(f'{label} Distribution (n={len(vals)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lactate_vs_bp(df, filename='lactate_vs_bp.png'):
    """Scatter plots of lactate vs SBP/DBP."""
    valid = df[(df['lactate_mean'] > 0) & (df['low_blood_pressure'] > 0) & (df['high_blood_pressure'] > 0)].copy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    bp_cols = [
        ('low_blood_pressure', 'DBP (mmHg)'),
        ('high_blood_pressure', 'SBP (mmHg)'),
    ]
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for row, (bp_col, bp_label) in enumerate(bp_cols):
        for col, lac_label in zip(lactate_cols, lactate_labels):
            ax = axes[row, lactate_cols.index(col)]
            
            x = valid[col].values
            y = valid[bp_col].values
            
            ax.scatter(x, y, alpha=0.4, s=20)
            
            # Linear fit
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 5:
                z = np.polyfit(x[mask], y[mask], 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x[mask])
                ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
                
                r, p_val = pearsonr(x[mask], y[mask])
                rho, _ = spearmanr(x[mask], y[mask])
                ax.set_title(f'{lac_label} vs {bp_label}\nPearson r={r:.3f}, p={p_val:.2e}\nSpearman ρ={rho:.3f}')
            else:
                ax.set_title(f'{lac_label} vs {bp_label}')
            
            ax.set_xlabel(lac_label)
            ax.set_ylabel(bp_label)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lactate_vs_age(df, filename='lactate_vs_age.png'):
    """Scatter plot of lactate vs age, colored by gender."""
    valid = df[(df['lactate_mean'] > 0) & (df['age'] > 0)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for ax, col, label in zip(axes, lactate_cols, lactate_labels):
        x = valid['age'].values
        y = valid[col].values
        
        # Color by gender if available
        if 'gender' in valid.columns:
            for gender, color, marker in [('男', 'blue', 'o'), ('女', 'red', '^')]:
                mask = valid['gender'] == gender
                if mask.sum() > 0:
                    ax.scatter(x[mask], y[mask], alpha=0.5, s=20, 
                              c=color, marker=marker, label=gender)
        
        # Overall fit
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 5:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x[mask])
            ax.plot(x_sorted, p(x_sorted), 'k--', linewidth=2)
            
            r, p_val = pearsonr(x[mask], y[mask])
            ax.set_title(f'{label} vs Age\nPearson r={r:.3f}, p={p_val:.2e}')
        
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lactate_by_gender(df, filename='lactate_by_gender.png'):
    """Box plot of lactate by gender."""
    valid = df[(df['lactate_mean'] > 0) & (df['gender'].isin(['男', '女']))].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for ax, col, label in zip(axes, lactate_cols, lactate_labels):
        data_male = valid[valid['gender'] == '男'][col].dropna()
        data_female = valid[valid['gender'] == '女'][col].dropna()
        
        bp = ax.boxplot([data_male, data_female], labels=['Male', 'Female'], 
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightpink')
        
        # Add individual points
        for i, data in enumerate([data_male, data_female]):
            jitter = np.random.normal(0, 0.05, len(data))
            ax.scatter(np.ones(len(data)) * (i + 1) + jitter, data, 
                      alpha=0.3, s=10, color='gray')
        
        # Statistical test
        if len(data_male) > 5 and len(data_female) > 5:
            stat, p_val = mannwhitneyu(data_male, data_female)
            ax.set_title(f'{label}\nMann-Whitney U p={p_val:.4f}')
        else:
            ax.set_title(label)
        
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lactate_vs_hr(df, filename='lactate_vs_hr.png'):
    """Scatter plot of lactate vs heart rate."""
    valid = df[(df['lactate_mean'] > 0) & (df['heart_rate'] > 0)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for ax, col, label in zip(axes, lactate_cols, lactate_labels):
        x = valid['heart_rate'].values
        y = valid[col].values
        
        ax.scatter(x, y, alpha=0.4, s=20)
        
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 5:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x[mask])
            ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
            
            r, p_val = pearsonr(x[mask], y[mask])
            ax.set_title(f'{label} vs Heart Rate\nPearson r={r:.3f}, p={p_val:.2e}')
        
        ax.set_xlabel('Heart Rate (bpm)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_lactate_vs_spo2(df, filename='lactate_vs_spo2.png'):
    """Scatter plot of lactate vs blood oxygen."""
    valid = df[(df['lactate_mean'] > 0) & (df['blood_oxygen'] > 0)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for ax, col, label in zip(axes, lactate_cols, lactate_labels):
        x = valid['blood_oxygen'].values
        y = valid[col].values
        
        ax.scatter(x, y, alpha=0.4, s=20)
        
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 5:
            r, p_val = pearsonr(x[mask], y[mask])
            ax.set_title(f'{label} vs SpO2\nPearson r={r:.3f}, p={p_val:.2e}')
        
        ax.set_xlabel('Blood Oxygen / SpO2 (%)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_matrix(df, filename='correlation_matrix.png'):
    """Plot correlation matrix of all numeric features."""
    # Select relevant numeric columns
    cols = ['age', 'low_blood_pressure', 'high_blood_pressure', 
            'heart_rate', 'blood_oxygen', 'respiratory_rate', 'temperature',
            'lactate_min', 'lactate_max', 'lactate_mean', 'lactate_median', 'lactate_count']
    
    valid = df[cols].copy()
    valid = valid.replace(-1, np.nan)
    
    # Compute correlation matrix
    corr = valid.corr(method='pearson')
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Labels
    labels = ['Age', 'DBP', 'SBP', 'HR', 'SpO2', 'RR', 'Temp',
              'Lac Min', 'Lac Max', 'Lac Mean', 'Lac Median', 'Lac Count']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=8)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Pearson Correlation Matrix', fontsize=14)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    
    # Also print the correlation table
    print("\n  Correlation Matrix (Pearson r):")
    print(f"  {'':>12}", end='')
    for l in labels:
        print(f"{l:>10}", end='')
    print()
    for i, l in enumerate(labels):
        print(f"  {l:>12}", end='')
        for j in range(len(labels)):
            print(f"{corr.values[i, j]:>10.3f}", end='')
        print()
    
    return corr


def plot_lactate_threshold_analysis(df, filename='lactate_threshold_analysis.png'):
    """Analyze BP differences between high and normal lactate groups."""
    valid = df[(df['lactate_mean'] > 0) & (df['low_blood_pressure'] > 0) & (df['high_blood_pressure'] > 0)].copy()
    
    # Define thresholds
    thresholds = np.linspace(1.0, 3.0, 21)
    
    results = []
    for thresh in thresholds:
        high_lac = valid[valid['lactate_mean'] >= thresh]
        normal_lac = valid[valid['lactate_mean'] < thresh]
        
        if len(high_lac) < 5 or len(normal_lac) < 5:
            continue
        
        sbp_diff = high_lac['high_blood_pressure'].mean() - normal_lac['high_blood_pressure'].mean()
        dbp_diff = high_lac['low_blood_pressure'].mean() - normal_lac['low_blood_pressure'].mean()
        
        _, sbp_p = ttest_ind(high_lac['high_blood_pressure'], normal_lac['high_blood_pressure'])
        _, dbp_p = ttest_ind(high_lac['low_blood_pressure'], normal_lac['low_blood_pressure'])
        
        results.append({
            'threshold': thresh,
            'n_high': len(high_lac),
            'n_normal': len(normal_lac),
            'sbp_diff': sbp_diff,
            'dbp_diff': dbp_diff,
            'sbp_p': sbp_p,
            'dbp_p': dbp_p,
        })
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SBP difference
    ax = axes[0, 0]
    ax.plot(results_df['threshold'], results_df['sbp_diff'], 'o-', color='red')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Lactate Mean Threshold (mmol/L)')
    ax.set_ylabel('SBP Difference (mmHg)')
    ax.set_title('SBP: High Lactate - Normal Lactate')
    ax.grid(True, alpha=0.3)
    
    # DBP difference
    ax = axes[0, 1]
    ax.plot(results_df['threshold'], results_df['dbp_diff'], 'o-', color='blue')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel('Lactate Mean Threshold (mmol/L)')
    ax.set_ylabel('DBP Difference (mmHg)')
    ax.set_title('DBP: High Lactate - Normal Lactate')
    ax.grid(True, alpha=0.3)
    
    # P-values
    ax = axes[1, 0]
    ax.plot(results_df['threshold'], results_df['sbp_p'], 'o-', color='red', label='SBP')
    ax.plot(results_df['threshold'], results_df['dbp_p'], 'o-', color='blue', label='DBP')
    ax.axhline(0.05, color='gray', linestyle='--', label='p=0.05')
    ax.set_yscale('log')
    ax.set_xlabel('Lactate Mean Threshold (mmol/L)')
    ax.set_ylabel('p-value (log scale)')
    ax.set_title('T-test p-values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Sample sizes
    ax = axes[1, 1]
    ax.plot(results_df['threshold'], results_df['n_high'], 'o-', color='orange', label='High Lactate')
    ax.plot(results_df['threshold'], results_df['n_normal'], 'o-', color='green', label='Normal Lactate')
    ax.set_xlabel('Lactate Mean Threshold (mmol/L)')
    ax.set_ylabel('Sample Size')
    ax.set_title('Group Sizes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    
    # Print best threshold
    best_idx = results_df['sbp_p'].idxmin()
    best = results_df.loc[best_idx]
    print(f"\n  Best lactate threshold for SBP differentiation:")
    print(f"    Threshold: {best['threshold']:.2f} mmol/L")
    print(f"    SBP diff: {best['sbp_diff']:.2f} mmHg (p={best['sbp_p']:.4f})")
    print(f"    DBP diff: {best['dbp_diff']:.2f} mmHg (p={best['dbp_p']:.4f})")
    print(f"    High lactate group: n={int(best['n_high'])}")
    print(f"    Normal lactate group: n={int(best['n_normal'])}")


def plot_lactate_vs_respiratory_rate(df, filename='lactate_vs_resp.png'):
    """Scatter plot of lactate vs respiratory rate."""
    valid = df[(df['lactate_mean'] > 0) & (df['respiratory_rate'] > 0)].copy()
    
    if len(valid) < 10:
        print(f"  Not enough data for lactate vs respiratory rate (n={len(valid)})")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    lactate_cols = ['lactate_min', 'lactate_mean', 'lactate_max']
    lactate_labels = ['Lactate Min (mmol/L)', 'Lactate Mean (mmol/L)', 'Lactate Max (mmol/L)']
    
    for ax, col, label in zip(axes, lactate_cols, lactate_labels):
        x = valid['respiratory_rate'].values
        y = valid[col].values
        
        ax.scatter(x, y, alpha=0.4, s=20)
        
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 5:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            x_sorted = np.sort(x[mask])
            ax.plot(x_sorted, p(x_sorted), 'r--', linewidth=2)
            
            r, p_val = pearsonr(x[mask], y[mask])
            ax.set_title(f'{label} vs Respiratory Rate\nPearson r={r:.3f}, p={p_val:.2e}')
        
        ax.set_xlabel('Respiratory Rate (bpm)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_summary_stats(df, corr):
    """Print a comprehensive summary of findings."""
    valid_lac = df[df['lactate_mean'] > 0].copy()
    valid_bp = valid_lac[(valid_lac['low_blood_pressure'] > 0) & (valid_lac['high_blood_pressure'] > 0)]
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Lactate Analysis Results")
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
                bp_name = 'DBP' if 'low' in bp_col else 'SBP'
                print(f"     {lac_col} vs {bp_name}: r={r:.3f}, p={p:.4f}")
    
    print(f"\n  3. Lactate vs Age:")
    valid_age = valid_lac[valid_lac['age'] > 0]
    if len(valid_age) > 5:
        r, p = pearsonr(valid_age['lactate_mean'], valid_age['age'])
        print(f"     Lactate mean vs Age: r={r:.3f}, p={p:.4f}")
    
    print(f"\n  4. Lactate by Gender:")
    valid_gender = valid_lac[valid_lac['gender'].isin(['男', '女'])]
    for gender in ['男', '女']:
        data = valid_gender[valid_gender['gender'] == gender]['lactate_mean'].dropna()
        gender_label = 'Male' if gender == '男' else 'Female'
        print(f"     {gender_label}: mean={data.mean():.2f}, median={data.median():.2f}, n={len(data)}")
    
    print(f"\n  5. Lactate vs Heart Rate:")
    valid_hr = valid_lac[valid_lac['heart_rate'] > 0]
    if len(valid_hr) > 5:
        r, p = pearsonr(valid_hr['lactate_mean'], valid_hr['heart_rate'])
        print(f"     Lactate mean vs HR: r={r:.3f}, p={p:.4f}")
    
    print(f"\n  6. Lactate vs SpO2:")
    valid_spo2 = valid_lac[valid_lac['blood_oxygen'] > 0]
    if len(valid_spo2) > 5:
        r, p = pearsonr(valid_spo2['lactate_mean'], valid_spo2['blood_oxygen'])
        print(f"     Lactate mean vs SpO2: r={r:.3f}, p={p:.4f}")
    
    print(f"\n  7. Key Correlations (from matrix):")
    lac_cols = ['lactate_min', 'lactate_mean', 'lactate_max', 'lactate_median']
    for lac in lac_cols:
        if lac in corr.columns:
            row = corr.loc[lac].drop(lac)
            top = row.abs().sort_values(ascending=False).head(3)
            print(f"     {lac} top correlates:")
            for idx, val in top.items():
                print(f"       {idx}: r={corr.loc[lac, idx]:.3f}")


def main():
    print("=" * 60)
    print("  Lactate Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_all_lab_data()
    
    # Filter valid lactate
    print("\n[2] Filtering valid lactate data...")
    valid = filter_valid_lactate(df)
    
    # Basic stats
    print("\n[3] Basic statistics...")
    basic_stats(valid, "Lactate Statistics (All Mirrors)")
    
    # Visualizations
    print("\n[4] Generating visualizations...")
    
    print("\n  4.1 Lactate distribution...")
    plot_lactate_distribution(valid)
    
    print("\n  4.2 Lactate vs BP...")
    plot_lactate_vs_bp(valid)
    
    print("\n  4.3 Lactate vs Age...")
    plot_lactate_vs_age(valid)
    
    print("\n  4.4 Lactate by Gender...")
    plot_lactate_by_gender(valid)
    
    print("\n  4.5 Lactate vs Heart Rate...")
    plot_lactate_vs_hr(valid)
    
    print("\n  4.6 Lactate vs SpO2...")
    plot_lactate_vs_spo2(valid)
    
    print("\n  4.7 Correlation matrix...")
    corr = plot_correlation_matrix(valid)
    
    print("\n  4.8 Lactate threshold analysis...")
    plot_lactate_threshold_analysis(valid)
    
    print("\n  4.9 Lactate vs Respiratory Rate...")
    plot_lactate_vs_respiratory_rate(valid)
    
    # Summary
    print("\n[5] Summary...")
    print_summary_stats(valid, corr)
    
    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
