import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import os

class SinglePatientBPPTTVisualizer:
    def __init__(self, csv_paths, min_measurements=2):
        # Accept either a single path or a list of paths
        if isinstance(csv_paths, str):
            self.csv_paths = [csv_paths]
        else:
            self.csv_paths = csv_paths
        self.min_measurements = min_measurements
        self.df = None
        self.patient_data = {}  # Dictionary of Hospital_Patient_ID -> dataframe
        self.patient_ids = []   # List of unique Hospital_Patient_IDs
        self.current_index = 0
        
        self.load_data()
        
        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.15, hspace=0.3)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.update_plots()
        
    def load_data(self):
        """Load and merge cleaned patient info from multiple CSVs, grouping by Hospital_Patient_ID."""
        # Load all CSV files
        all_dfs = []
        for csv_path in self.csv_paths:
            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            all_dfs.append(df)
        
        if not all_dfs:
            raise FileNotFoundError(f"No valid CSV files found in: {self.csv_paths}")
        
        # Concatenate all dataframes
        self.df = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(all_dfs)} CSV file(s) with {len(self.df)} total records")
        
        # Validate required columns
        required_cols = ['Lab_Patient_ID', 'Hospital_Patient_ID', 'ECG_SQI_AVG', 
                        'rPPG_SQI_AVG', 'PTT', 'PTT_STDDEV', 'PTT_LENGTH',
                        'Low_Blood_Pressure', 'High_Blood_Pressure', 'HR_MEAN']
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with invalid blood pressure or PTT
        valid_df = self.df[
            (self.df['Low_Blood_Pressure'] > 0) & 
            (self.df['High_Blood_Pressure'] > 0) &
            (self.df['PTT'].notna())
        ].copy()
        
        # Calculate Mean BP
        valid_df['Mean_BP'] = (valid_df['Low_Blood_Pressure'] + valid_df['High_Blood_Pressure']) / 2
        
        # Group by Hospital_Patient_ID
        grouped = valid_df.groupby('Hospital_Patient_ID')
        
        # Only keep patients with sufficient data points (>= min_measurements)
        total_patients = len(grouped)
        for hospital_id, group_df in grouped:
            if len(group_df) >= self.min_measurements:
                self.patient_data[hospital_id] = group_df.copy()
        
        self.patient_ids = sorted(self.patient_data.keys())
        
        if not self.patient_ids:
            raise ValueError(f"No patients with at least {self.min_measurements} valid data points found!")
        
        filtered_out = total_patients - len(self.patient_ids)
        print(f"Loaded {len(self.patient_ids)} patients with at least {self.min_measurements} measurements")
        if filtered_out > 0:
            print(f"  (Filtered out {filtered_out} patients with < {self.min_measurements} measurements)")
        print(f"Total data points: {sum(len(df) for df in self.patient_data.values())}")
        
    def calculate_all_correlations(self):
        """Calculate correlation coefficients for all patients."""
        correlations = []
        
        for patient_id in self.patient_ids:
            patient_df = self.patient_data[patient_id]
            
            ptt = patient_df['PTT'].values
            low_bp = patient_df['Low_Blood_Pressure'].values
            high_bp = patient_df['High_Blood_Pressure'].values
            mean_bp = patient_df['Mean_BP'].values
            hr = patient_df['HR_MEAN'].values
            
            # Calculate correlations and residuals
            r_diastolic = None
            r_systolic = None
            r_mean = None
            r_hr = None
            me_diastolic = None
            std_diastolic = None
            me_systolic = None
            std_systolic = None
            me_mean = None
            std_mean = None
            me_hr = None
            std_hr = None
            
            if len(ptt) >= 2:
                try:
                    slope_dia, intercept_dia, r_diastolic, _, _ = stats.linregress(ptt, low_bp)
                    y_pred_dia = slope_dia * ptt + intercept_dia
                    residuals_dia = abs(y_pred_dia - low_bp)
                    me_diastolic = np.mean(residuals_dia)
                    std_diastolic = np.std(residuals_dia, ddof=1)
                    
                    slope_sys, intercept_sys, r_systolic, _, _ = stats.linregress(ptt, high_bp)
                    y_pred_sys = slope_sys * ptt + intercept_sys
                    residuals_sys = abs(y_pred_sys - high_bp)
                    me_systolic = np.mean(residuals_sys)
                    std_systolic = np.std(residuals_sys, ddof=1)
                    
                    slope_mean, intercept_mean, r_mean, _, _ = stats.linregress(ptt, mean_bp)
                    y_pred_mean = slope_mean * ptt + intercept_mean
                    residuals_mean = abs(y_pred_mean - mean_bp)
                    me_mean = np.mean(residuals_mean)
                    std_mean = np.std(residuals_mean, ddof=1)
                    
                    # HR vs PTT correlation (filter out NaN values)
                    valid_hr_mask = ~np.isnan(hr)
                    if np.sum(valid_hr_mask) >= 2:
                        ptt_valid = ptt[valid_hr_mask]
                        hr_valid = hr[valid_hr_mask]
                        slope_hr, intercept_hr, r_hr, _, _ = stats.linregress(ptt_valid, hr_valid)
                        y_pred_hr = slope_hr * ptt_valid + intercept_hr
                        residuals_hr = abs(y_pred_hr - hr_valid)
                        me_hr = np.mean(residuals_hr)
                        std_hr = np.std(residuals_hr, ddof=1)
                except:
                    pass
            
            correlations.append({
                'patient_id': patient_id,
                'n_measurements': len(patient_df),
                'r_diastolic': r_diastolic,
                'r_systolic': r_systolic,
                'r_mean': r_mean,
                'r_hr': r_hr,
                'me_diastolic': me_diastolic,
                'std_diastolic': std_diastolic,
                'me_systolic': me_systolic,
                'std_systolic': std_systolic,
                'me_mean': me_mean,
                'std_mean': std_mean,
                'me_hr': me_hr,
                'std_hr': std_hr,
                'ecg_sqi_avg': patient_df['ECG_SQI_AVG'].mean(),
                'rppg_sqi_avg': patient_df['rPPG_SQI_AVG'].mean(),
                'ptt_stddev_avg': patient_df['PTT_STDDEV'].mean()
            })
        
        return pd.DataFrame(correlations)
    
    def show_correlation_summary(self):
        """Display summary of all patient correlations."""
        corr_df = self.calculate_all_correlations()
        
        # Create summary figure
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle('Correlation Coefficient Summary - All Patients', 
                    fontsize=16, fontweight='bold')
        
        # Histograms of correlation coefficients
        ax = axes[0, 0]
        valid_r_dia = corr_df['r_diastolic'].dropna()
        ax.hist(valid_r_dia, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(valid_r_dia.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {valid_r_dia.mean():.3f}')
        ax.set_xlabel('Correlation Coefficient (R)', fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=11)
        ax.set_title('PTT vs Diastolic BP', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        valid_r_sys = corr_df['r_systolic'].dropna()
        ax.hist(valid_r_sys, bins=20, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(valid_r_sys.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {valid_r_sys.mean():.3f}')
        ax.set_xlabel('Correlation Coefficient (R)', fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=11)
        ax.set_title('PTT vs Systolic BP', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        valid_r_mean = corr_df['r_mean'].dropna()
        ax.hist(valid_r_mean, bins=20, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax.axvline(valid_r_mean.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {valid_r_mean.mean():.3f}')
        ax.set_xlabel('Correlation Coefficient (R)', fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=11)
        ax.set_title('PTT vs Mean BP', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 3]
        valid_r_hr = corr_df['r_hr'].dropna()
        ax.hist(valid_r_hr, bins=20, alpha=0.7, color='orchid', edgecolor='black')
        ax.axvline(valid_r_hr.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {valid_r_hr.mean():.3f}')
        ax.set_xlabel('Correlation Coefficient (R)', fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=11)
        ax.set_title('PTT vs HR', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Scatter plots: Number of measurements vs correlation
        ax = axes[1, 0]
        ax.scatter(corr_df['n_measurements'], corr_df['r_diastolic'], 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Number of Measurements', fontsize=11)
        ax.set_ylabel('R (Diastolic)', fontsize=11)
        ax.set_title('Measurements vs Correlation (Diastolic)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.scatter(corr_df['n_measurements'], corr_df['r_systolic'],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='coral')
        ax.set_xlabel('Number of Measurements', fontsize=11)
        ax.set_ylabel('R (Systolic)', fontsize=11)
        ax.set_title('Measurements vs Correlation (Systolic)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        ax.scatter(corr_df['n_measurements'], corr_df['r_hr'],
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5, color='orchid')
        ax.set_xlabel('Number of Measurements', fontsize=11)
        ax.set_ylabel('R (HR)', fontsize=11)
        ax.set_title('Measurements vs Correlation (HR)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Summary statistics text
        ax = axes[1, 3]
        ax.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
        summary_text += f"Total Patients: {len(corr_df)}\n\n"
        
        summary_text += "Diastolic BP Correlation:\n"
        summary_text += f"  Mean R: {valid_r_dia.mean():.4f}\n"
        summary_text += f"  Median R: {valid_r_dia.median():.4f}\n"
        summary_text += f"  Std R: {valid_r_dia.std():.4f}\n"
        summary_text += f"  Range: [{valid_r_dia.min():.3f}, {valid_r_dia.max():.3f}]\n\n"
        
        summary_text += "Systolic BP Correlation:\n"
        summary_text += f"  Mean R: {valid_r_sys.mean():.4f}\n"
        summary_text += f"  Median R: {valid_r_sys.median():.4f}\n"
        summary_text += f"  Std R: {valid_r_sys.std():.4f}\n"
        summary_text += f"  Range: [{valid_r_sys.min():.3f}, {valid_r_sys.max():.3f}]\n\n"
        
        summary_text += "Mean BP Correlation:\n"
        summary_text += f"  Mean R: {valid_r_mean.mean():.4f}\n"
        summary_text += f"  Median R: {valid_r_mean.median():.4f}\n"
        summary_text += f"  Std R: {valid_r_mean.std():.4f}\n"
        summary_text += f"  Range: [{valid_r_mean.min():.3f}, {valid_r_mean.max():.3f}]\n\n"
        
        summary_text += "HR Correlation:\n"
        summary_text += f"  Mean R: {valid_r_hr.mean():.4f}\n"
        summary_text += f"  Median R: {valid_r_hr.median():.4f}\n"
        summary_text += f"  Std R: {valid_r_hr.std():.4f}\n"
        summary_text += f"  Range: [{valid_r_hr.min():.3f}, {valid_r_hr.max():.3f}]\n\n"
        
        summary_text += "ME (mean) ± STD (worst):\n"
        valid_me_dia = corr_df['me_diastolic'].dropna()
        valid_std_dia = corr_df['std_diastolic'].dropna()
        valid_me_sys = corr_df['me_systolic'].dropna()
        valid_std_sys = corr_df['std_systolic'].dropna()
        valid_me_mean = corr_df['me_mean'].dropna()
        valid_std_mean = corr_df['std_mean'].dropna()
        valid_me_hr = corr_df['me_hr'].dropna()
        valid_std_hr = corr_df['std_hr'].dropna()
        summary_text += f"  Diastolic: {valid_me_dia.mean():.2f}±{valid_std_dia.max():.2f}\n"
        summary_text += f"  Systolic: {valid_me_sys.mean():.2f}±{valid_std_sys.max():.2f}\n"
        summary_text += f"  Mean BP: {valid_me_mean.mean():.2f}±{valid_std_mean.max():.2f}\n"
        summary_text += f"  HR: {valid_me_hr.mean():.2f}±{valid_std_hr.max():.2f} BPM\n\n"
        
        summary_text += f"Avg Measurements/Patient: {corr_df['n_measurements'].mean():.1f}\n"
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add instruction at bottom
        fig.text(0.5, 0.02, 'Press any key to continue to individual patient view...', 
                ha='center', fontsize=12, style='italic', color='darkblue',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        # Print to console
        print(f"\n{'='*70}")
        print("CORRELATION SUMMARY - ALL PATIENTS")
        print(f"{'='*70}")
        print(f"Total Patients: {len(corr_df)}")
        print(f"\nDiastolic BP Correlation:")
        print(f"  Mean R: {valid_r_dia.mean():.4f} ± {valid_r_dia.std():.4f}")
        print(f"  Median R: {valid_r_dia.median():.4f}")
        print(f"  Range: [{valid_r_dia.min():.3f}, {valid_r_dia.max():.3f}]")
        print(f"\nSystolic BP Correlation:")
        print(f"  Mean R: {valid_r_sys.mean():.4f} ± {valid_r_sys.std():.4f}")
        print(f"  Median R: {valid_r_sys.median():.4f}")
        print(f"  Range: [{valid_r_sys.min():.3f}, {valid_r_sys.max():.3f}]")
        print(f"\nMean BP Correlation:")
        print(f"  Mean R: {valid_r_mean.mean():.4f} ± {valid_r_mean.std():.4f}")
        print(f"  Median R: {valid_r_mean.median():.4f}")
        print(f"  Range: [{valid_r_mean.min():.3f}, {valid_r_mean.max():.3f}]")
        
        print(f"\nHR Correlation:")
        print(f"  Mean R: {valid_r_hr.mean():.4f} ± {valid_r_hr.std():.4f}")
        print(f"  Median R: {valid_r_hr.median():.4f}")
        print(f"  Range: [{valid_r_hr.min():.3f}, {valid_r_hr.max():.3f}]")
        
        valid_me_dia = corr_df['me_diastolic'].dropna()
        valid_std_dia = corr_df['std_diastolic'].dropna()
        valid_me_sys = corr_df['me_systolic'].dropna()
        valid_std_sys = corr_df['std_systolic'].dropna()
        valid_me_mean = corr_df['me_mean'].dropna()
        valid_std_mean = corr_df['std_mean'].dropna()
        valid_me_hr = corr_df['me_hr'].dropna()
        valid_std_hr = corr_df['std_hr'].dropna()
        print(f"\nME (mean) ± STD (worst case):")
        print(f"  Diastolic BP: {valid_me_dia.mean():.2f}±{valid_std_dia.max():.2f} mmHg")
        print(f"  Systolic BP:  {valid_me_sys.mean():.2f}±{valid_std_sys.max():.2f} mmHg")
        print(f"  Mean BP:      {valid_me_mean.mean():.2f}±{valid_std_mean.max():.2f} mmHg")
        print(f"  HR:           {valid_me_hr.mean():.2f}±{valid_std_hr.max():.2f} BPM")
        print(f"\nPress any key in the plot window to continue...")
        print(f"{'='*70}\n")
        
        # Wait for key press
        fig.canvas.mpl_connect('key_press_event', lambda event: plt.close(fig))
        plt.show()
        
        return corr_df
    
    def on_key_press(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.patient_ids)
            self.update_plots()
        elif event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.patient_ids)
            self.update_plots()
        elif event.key == 'escape':
            plt.close(self.fig)
            
    def plot_regression(self, ax, x, y, xlabel, ylabel, title, patient_id):
        """Plot scatter with regression line and stats."""
        ax.clear()
        
        if len(x) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return None
        
        # Scatter plot with larger markers for single patient view
        ax.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidth=1.5, c='steelblue')
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2.5, label=f'y = {slope:.4f}x + {intercept:.2f}')
        
        # Calculate residuals: ME = mean(predicted - reference)
        y_pred = slope * x + intercept
        residuals = abs(y_pred - y)
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals, ddof=1)  # Use sample std (n-1)
        
        # Add statistics to plot
        stats_text = f'R = {r_value:.4f}\n'
        stats_text += f'R² = {r_value**2:.4f}\n'
        stats_text += f'p = {p_value:.4e}\n'
        stats_text += f'n = {len(x)}\n'
        stats_text += f'ME±STD = {residual_mean:.2f}±{residual_std:.2f}'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return r_value, residual_mean, residual_std
        
    def update_plots(self):
        """Update all three plots for current patient."""
        if not self.patient_ids:
            return
        
        # Get current patient data
        current_patient_id = self.patient_ids[self.current_index]
        patient_df = self.patient_data[current_patient_id]
        
        # Extract data
        ptt = patient_df['PTT'].values
        low_bp = patient_df['Low_Blood_Pressure'].values
        high_bp = patient_df['High_Blood_Pressure'].values
        mean_bp = patient_df['Mean_BP'].values
        
        # Get patient statistics
        ecg_sqi_avg = patient_df['ECG_SQI_AVG'].mean()
        rppg_sqi_avg = patient_df['rPPG_SQI_AVG'].mean()
        ptt_stddev_avg = patient_df['PTT_STDDEV'].mean()
        ptt_length_total = patient_df['PTT_LENGTH'].sum()
        lab_ids = patient_df['Lab_Patient_ID'].values
        
        # Plot 1: PTT vs Diastolic BP
        result1 = self.plot_regression(
            self.axes[0], ptt, low_bp,
            'PTT (s)', 'Diastolic BP (mmHg)',
            'PTT vs Diastolic BP',
            current_patient_id
        )
        r1 = result1[0] if result1 else None
        r1_mean = result1[1] if result1 else None
        r1_std = result1[2] if result1 else None
        
        # Plot 2: PTT vs Systolic BP
        result2 = self.plot_regression(
            self.axes[1], ptt, high_bp,
            'PTT (s)', 'Systolic BP (mmHg)',
            'PTT vs Systolic BP',
            current_patient_id
        )
        r2 = result2[0] if result2 else None
        r2_mean = result2[1] if result2 else None
        r2_std = result2[2] if result2 else None
        
        # Plot 3: PTT vs Mean BP
        result3 = self.plot_regression(
            self.axes[2], ptt, mean_bp,
            'PTT (s)', 'Mean BP (mmHg)',
            'PTT vs Mean BP',
            current_patient_id
        )
        r3 = result3[0] if result3 else None
        r3_mean = result3[1] if result3 else None
        r3_std = result3[2] if result3 else None
        
        # Update main title with patient info
        title_text = f'Patient: {current_patient_id} ({self.current_index + 1}/{len(self.patient_ids)}) | '
        title_text += f'{len(patient_df)} measurements\n'
        title_text += f'Lab IDs: {", ".join(map(str, lab_ids))} | '
        title_text += f'Avg ECG SQI: {ecg_sqi_avg:.3f} | Avg rPPG SQI: {rppg_sqi_avg:.3f} | '
        title_text += f'Avg PTT StdDev: {ptt_stddev_avg:.4f} | Total PTT Length: {ptt_length_total}'
        
        self.fig.suptitle(
            title_text,
            fontsize=11, fontweight='bold'
        )
        
        # Add navigation instructions
        nav_text = 'Use ← → arrow keys to navigate between patients | Press ESC to exit'
        self.fig.text(0.5, 0.02, nav_text, ha='center', fontsize=10, 
                     style='italic', color='darkblue')
        
        self.fig.canvas.draw_idle()
        
        # Print summary statistics to console
        print(f"\n{'='*70}")
        print(f"Patient: {current_patient_id} ({self.current_index + 1}/{len(self.patient_ids)})")
        print(f"{'='*70}")
        print(f"Measurements: {len(patient_df)}")
        print(f"Lab Patient IDs: {', '.join(map(str, lab_ids))}")
        print(f"\nAverage Quality Metrics:")
        print(f"  ECG SQI: {ecg_sqi_avg:.4f}")
        print(f"  rPPG SQI: {rppg_sqi_avg:.4f}")
        print(f"  PTT StdDev: {ptt_stddev_avg:.4f}")
        print(f"  Total PTT Length: {ptt_length_total}")
        print(f"\nPTT Range: {ptt.min():.4f}s - {ptt.max():.4f}s (mean: {ptt.mean():.4f}s)")
        print(f"Diastolic BP Range: {low_bp.min():.0f} - {low_bp.max():.0f} mmHg")
        print(f"Systolic BP Range: {high_bp.min():.0f} - {high_bp.max():.0f} mmHg")
        
        if r1 is not None:
            print(f"\nCorrelation Coefficients & Residuals:")
            print(f"  PTT vs Diastolic BP: R = {r1:.4f}, R² = {r1**2:.4f}, ME±STD = {r1_mean:.2f}±{r1_std:.2f} mmHg")
        if r2 is not None:
            print(f"  PTT vs Systolic BP:  R = {r2:.4f}, R² = {r2**2:.4f}, ME±STD = {r2_mean:.2f}±{r2_std:.2f} mmHg")
        if r3 is not None:
            print(f"  PTT vs Mean BP:      R = {r3:.4f}, R² = {r3**2:.4f}, ME±STD = {r3_mean:.2f}±{r3_std:.2f} mmHg")
        print(f"{'='*70}\n")
        
    def show(self):
        """Display the interactive plot."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize per-patient PTT vs Blood Pressure relationship'
    )
    parser.add_argument(
        '--csv', 
        type=str,
        nargs='+',
        default=['./mirror2_auto_cleaned/cleaned_patient_info.csv', './mirror1_auto_cleaned/cleaned_patient_info.csv'],
        help='Path(s) to cleaned_patient_info.csv file(s). Can specify multiple files.'
    )
    parser.add_argument(
        '--min_measurements',
        type=int,
        default=3,
        help='Minimum number of measurements required per patient (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer with potentially multiple CSV files
    visualizer = SinglePatientBPPTTVisualizer(args.csv, min_measurements=args.min_measurements)
    
    # First show correlation summary for all patients
    print("\nGenerating correlation summary for all patients...")
    visualizer.show_correlation_summary()
    
    # Then show individual patient view
    print("\nStarting individual patient view...")
    print("Use ← → arrow keys to navigate, ESC to exit\n")
    visualizer.show()


if __name__ == '__main__':
    main()
