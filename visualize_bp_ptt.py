import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import stats
import argparse
import os

class BPPTTVisualizer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
        
        # Default thresholds
        self.thresholds = {
            'ptt_stddev_max': 0.1,
            'ecg_sqi_min': 0.7,
            'rppg_sqi_min': 0.8,
            'ptt_length_min': 10
        }
        
        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(bottom=0.35, hspace=0.3)
        
        # Create sliders
        self.setup_sliders()
        
        # Initial plot
        self.update_plots()
        
    def load_data(self):
        """Load the cleaned patient info CSV."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        required_cols = ['Lab_Patient_ID', 'Hospital_Patient_ID', 'ECG_SQI_AVG', 
                        'rPPG_SQI_AVG', 'PTT', 'PTT_STDDEV', 'PTT_LENGTH',
                        'Low_Blood_Pressure', 'High_Blood_Pressure']
        
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with invalid blood pressure (-1 or NaN)
        self.df = self.df[
            (self.df['Low_Blood_Pressure'] > 0) & 
            (self.df['High_Blood_Pressure'] > 0) &
            (self.df['PTT'].notna())
        ].copy()
        
        # Calculate Mean Arterial Pressure (MAP)
        # MAP = (2*SBP + DBP) / 3 is a common approximation
        # Or simpler: (SBP + DBP) / 2
        self.df['Mean_BP'] = (self.df['Low_Blood_Pressure'] + self.df['High_Blood_Pressure']) / 2
        
        print(f"Loaded {len(self.df)} patients with valid blood pressure data")
        
    def setup_sliders(self):
        """Create interactive sliders for threshold adjustment."""
        # Slider axes
        ax_ptt_std = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_ecg_sqi = plt.axes([0.15, 0.15, 0.7, 0.02])
        ax_rppg_sqi = plt.axes([0.15, 0.10, 0.7, 0.02])
        ax_ptt_len = plt.axes([0.15, 0.05, 0.7, 0.02])
        
        # Create sliders
        self.slider_ptt_std = Slider(
            ax_ptt_std, 'Max PTT StdDev', 0.0, 0.15, 
            valinit=self.thresholds['ptt_stddev_max'], valstep=0.005
        )
        
        self.slider_ecg_sqi = Slider(
            ax_ecg_sqi, 'Min ECG SQI', 0.0, 1.0, 
            valinit=self.thresholds['ecg_sqi_min'], valstep=0.05
        )
        
        self.slider_rppg_sqi = Slider(
            ax_rppg_sqi, 'Min rPPG SQI', 0.0, 1.0, 
            valinit=self.thresholds['rppg_sqi_min'], valstep=0.05
        )
        
        self.slider_ptt_len = Slider(
            ax_ptt_len, 'Min PTT Length', 0, 100, 
            valinit=self.thresholds['ptt_length_min'], valstep=1
        )
        
        # Connect sliders to update function
        self.slider_ptt_std.on_changed(self.on_slider_change)
        self.slider_ecg_sqi.on_changed(self.on_slider_change)
        self.slider_rppg_sqi.on_changed(self.on_slider_change)
        self.slider_ptt_len.on_changed(self.on_slider_change)
        
    def on_slider_change(self, val):
        """Update thresholds and replot."""
        self.thresholds['ptt_stddev_max'] = self.slider_ptt_std.val
        self.thresholds['ecg_sqi_min'] = self.slider_ecg_sqi.val
        self.thresholds['rppg_sqi_min'] = self.slider_rppg_sqi.val
        self.thresholds['ptt_length_min'] = self.slider_ptt_len.val
        self.update_plots()
        
    def filter_data(self):
        """Apply threshold filters to data."""
        filtered = self.df[
            (self.df['PTT_STDDEV'] <= self.thresholds['ptt_stddev_max']) &
            (self.df['ECG_SQI_AVG'] >= self.thresholds['ecg_sqi_min']) &
            (self.df['rPPG_SQI_AVG'] >= self.thresholds['rppg_sqi_min']) &
            (self.df['PTT_LENGTH'] >= self.thresholds['ptt_length_min'])
        ].copy()
        
        return filtered
        
    def plot_regression(self, ax, x, y, xlabel, ylabel, title):
        """Plot scatter with regression line and stats."""
        ax.clear()
        
        if len(x) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Regression line
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.4f}x + {intercept:.2f}')
        
        # Add statistics to plot
        stats_text = f'R = {r_value:.4f}\n'
        stats_text += f'R² = {r_value**2:.4f}\n'
        stats_text += f'p = {p_value:.4e}\n'
        stats_text += f'n = {len(x)}'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return r_value
        
    def update_plots(self):
        """Update all three plots with current thresholds."""
        filtered_df = self.filter_data()
        
        if len(filtered_df) == 0:
            for ax in self.axes:
                ax.clear()
                ax.text(0.5, 0.5, 'No data meets criteria', 
                       ha='center', va='center', transform=ax.transAxes)
            self.fig.suptitle(f'PTT vs Blood Pressure (0 patients after filtering)', 
                            fontsize=14, fontweight='bold')
            self.fig.canvas.draw_idle()
            return
        
        # Extract data
        ptt = filtered_df['PTT'].values
        low_bp = filtered_df['Low_Blood_Pressure'].values
        high_bp = filtered_df['High_Blood_Pressure'].values
        mean_bp = filtered_df['Mean_BP'].values
        
        # Plot 1: PTT vs Low Blood Pressure (Diastolic)
        r1 = self.plot_regression(
            self.axes[0], ptt, low_bp,
            'PTT (s)', 'Diastolic BP (mmHg)',
            'PTT vs Diastolic Blood Pressure'
        )
        
        # Plot 2: PTT vs High Blood Pressure (Systolic)
        r2 = self.plot_regression(
            self.axes[1], ptt, high_bp,
            'PTT (s)', 'Systolic BP (mmHg)',
            'PTT vs Systolic Blood Pressure'
        )
        
        # Plot 3: PTT vs Mean Blood Pressure
        r3 = self.plot_regression(
            self.axes[2], ptt, mean_bp,
            'PTT (s)', 'Mean BP (mmHg)',
            'PTT vs Mean Blood Pressure'
        )
        
        # Update main title with filter info
        self.fig.suptitle(
            f'PTT vs Blood Pressure ({len(filtered_df)}/{len(self.df)} patients | '
            f'PTT_STD≤{self.thresholds["ptt_stddev_max"]:.3f}, '
            f'ECG≥{self.thresholds["ecg_sqi_min"]:.2f}, '
            f'rPPG≥{self.thresholds["rppg_sqi_min"]:.2f}, '
            f'Length≥{int(self.thresholds["ptt_length_min"])})',
            fontsize=13, fontweight='bold'
        )
        
        self.fig.canvas.draw_idle()
        
        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"Filtered Data: {len(filtered_df)} / {len(self.df)} patients")
        print(f"{'='*60}")
        print(f"Thresholds:")
        print(f"  PTT StdDev ≤ {self.thresholds['ptt_stddev_max']:.3f}")
        print(f"  ECG SQI ≥ {self.thresholds['ecg_sqi_min']:.2f}")
        print(f"  rPPG SQI ≥ {self.thresholds['rppg_sqi_min']:.2f}")
        print(f"  PTT Length ≥ {int(self.thresholds['ptt_length_min'])}")
        print(f"\nCorrelation Coefficients:")
        print(f"  PTT vs Diastolic BP: R = {r1:.4f}, R² = {r1**2:.4f}")
        print(f"  PTT vs Systolic BP:  R = {r2:.4f}, R² = {r2**2:.4f}")
        print(f"  PTT vs Mean BP:      R = {r3:.4f}, R² = {r3**2:.4f}")
        print(f"{'='*60}\n")
        
    def show(self):
        """Display the interactive plot."""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PTT vs Blood Pressure relationship with interactive filtering'
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        default='./mirror2_auto_cleaned/cleaned_patient_info.csv',
        help='Path to cleaned_patient_info.csv file'
    )
    parser.add_argument(
        '--ptt_stddev_max', 
        type=float, 
        default=0.1,
        help='Maximum PTT standard deviation threshold (default: 0.1)'
    )
    parser.add_argument(
        '--ecg_sqi_min', 
        type=float, 
        default=0.7,
        help='Minimum ECG SQI threshold (default: 0.7)'
    )
    parser.add_argument(
        '--rppg_sqi_min', 
        type=float, 
        default=0.8,
        help='Minimum rPPG SQI threshold (default: 0.8)'
    )
    parser.add_argument(
        '--ptt_length_min', 
        type=int, 
        default=10,
        help='Minimum PTT length threshold (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BPPTTVisualizer(args.csv)
    
    # Override default thresholds with command line arguments
    visualizer.thresholds['ptt_stddev_max'] = args.ptt_stddev_max
    visualizer.thresholds['ecg_sqi_min'] = args.ecg_sqi_min
    visualizer.thresholds['rppg_sqi_min'] = args.rppg_sqi_min
    visualizer.thresholds['ptt_length_min'] = args.ptt_length_min
    
    # Update sliders to match
    visualizer.slider_ptt_std.set_val(args.ptt_stddev_max)
    visualizer.slider_ecg_sqi.set_val(args.ecg_sqi_min)
    visualizer.slider_rppg_sqi.set_val(args.rppg_sqi_min)
    visualizer.slider_ptt_len.set_val(args.ptt_length_min)
    
    # Show interactive plot
    visualizer.show()


if __name__ == '__main__':
    main()
