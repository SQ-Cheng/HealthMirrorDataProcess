import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib.widgets import Slider

import global_vars


CANONICAL_COLUMNS = {
    'Time': ['Time', 'time', 'timestamp', 'timestamps'],
    'rppg': ['rppg'],
    'ecg': ['ecg'],
    'ppg_red': ['ppg_red', 'red'],
    'ppg_ir': ['ppg_ir', 'ir'],
    'ppg_green': ['ppg_green', 'green']
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    df = df.rename(columns=rename_map)
    for column in CANONICAL_COLUMNS.keys():
        if column not in df:
            df[column] = 0.0
    return df[list(CANONICAL_COLUMNS.keys())]


class SignalData:
    ordered_columns = list(CANONICAL_COLUMNS.keys())

    def __init__(self, frame: pd.DataFrame | None = None):
        if frame is None:
            self.frame = pd.DataFrame({col: [] for col in self.ordered_columns})
        else:
            missing = [col for col in self.ordered_columns if col not in frame]
            for col in missing:
                frame[col] = 0.0
            self.frame = frame[self.ordered_columns].copy()

    @property
    def time(self) -> np.ndarray:
        return self.frame['Time'].to_numpy(copy=True)

    def get_signal(self, name: str) -> np.ndarray:
        if name not in self.frame:
            return np.array([])
        return self.frame[name].to_numpy(copy=True)

    def has_ppg(self) -> bool:
        return any(self.frame[col].abs().sum() > 0 for col in ['ppg_red', 'ppg_ir', 'ppg_green'])


class DataLoader:
    def __init__(self, version: str):
        self.version = version

    def load(self, data_path: str) -> SignalData:
        base_path = Path(data_path)
        csv_name = 'merged_log.csv' if self.version == '1' else 'normalized_log.csv'
        csv_path = base_path / csv_name
        if not csv_path.exists():
            return SignalData()

        df = pd.read_csv(csv_path)
        if 'timestamp' not in df.columns and 'Time' not in df.columns:
            if df.shape[1] >= 1:
                df = df.rename(columns={df.columns[0]: 'Time'})
        df = _standardize_columns(df)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return SignalData(df)


class SignalCleaner:
    def __init__(self, fs: int = 512):
        self.fs = fs

    def clean(self, sig, config) -> np.ndarray:
        array = np.asarray(sig, dtype=float)
        if array.size == 0:
            return np.array([], dtype=bool)

        mask = np.ones(array.size, dtype=bool)
        if 'std' in config:
            mask &= self._std_filter(array, config['std'])
        if 'diff' in config:
            mask &= self._diff_filter(array, config['diff'])
        if 'welch' in config:
            mask &= self._welch_filter(array, config['welch'])
        return mask

    def _std_filter(self, sig, params):
        window_len = max(1, int(params['window_size'] * self.fs))
        threshold = params['threshold']
        global_std = np.std(sig)
        mask = np.ones(sig.size, dtype=bool)
        for start in range(0, sig.size - window_len, window_len):
            if np.std(sig[start:start + window_len]) > global_std * threshold:
                mask[start:start + window_len] = False
        return mask

    def _diff_filter(self, sig, params):
        window_len = max(1, int(params['window_size'] * self.fs))
        threshold = params['threshold']
        global_diff = np.ptp(sig)
        mask = np.ones(sig.size, dtype=bool)
        for start in range(0, sig.size - window_len, window_len):
            seg = sig[start:start + window_len]
            if np.ptp(seg) > global_diff * threshold:
                mask[start:start + window_len] = False
        return mask

    def _welch_filter(self, sig, params):
        window_len = max(8, int(params['window_size'] * self.fs))
        freq_tolerance = params['bpm_tolerance'] / 60
        gf, gPxx = signal.welch(sig, fs=self.fs, nperseg=window_len)
        peak_freq = gf[np.argmax(gPxx)]
        mask = np.ones(sig.size, dtype=bool)
        for start in range(0, sig.size - window_len, window_len):
            seg = sig[start:start + window_len]
            f, Pxx = signal.welch(seg, fs=self.fs, nperseg=window_len)
            if abs(f[np.argmax(Pxx)] - peak_freq) > freq_tolerance:
                mask[start:start + window_len] = False
        return mask


class DataLogger:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

    def log_cleaned_data(self, file_name: str, data: SignalData, masks: dict[str, np.ndarray]):
        signal_names = ['rppg', 'ecg']
        if data.has_ppg():
            signal_names.extend(['ppg_red', 'ppg_ir', 'ppg_green'])

        combined_mask = np.ones(data.time.size, dtype=bool)
        for name in signal_names:
            if name in masks:
                combined_mask &= masks[name]

        for idx, (start, end) in enumerate(self._find_clean_windows(combined_mask)):
            if end - start <= 0:
                continue
            window_df = pd.DataFrame({'Time': data.time[start:end]})
            for name in signal_names:
                window_df[name] = data.get_signal(name)[start:end]
            output_name = file_name.replace('.csv', f'_{idx + 1}.csv')
            window_df.to_csv(self.log_path / output_name, index=False)

    def modify_cleaned_data(self, file_name: str, option: str):
        target_path = self.log_path / file_name
        if not target_path.exists():
            return

        if option == 'reject':
            target_path.unlink(missing_ok=True)
            return

        df = pd.read_csv(target_path)
        if option == 'reverse':
            ecg_cols = [c for c in df.columns if 'ecg' in c.lower()]
            df[ecg_cols] = df[ecg_cols] * -1

        numeric_cols = [c for c in df.columns if c != 'Time']
        df[numeric_cols] = df[numeric_cols].apply(lambda col: (col - col.mean()) / col.std(ddof=0))
        df.to_csv(target_path, index=False)

    @staticmethod
    def _find_clean_windows(mask: np.ndarray) -> list[tuple[int, int]]:
        windows = []
        start = 0
        while start < mask.size:
            if mask[start]:
                end = start
                while end < mask.size and mask[end]:
                    end += 1
                windows.append((start, end))
                start = end
            else:
                start += 1
        return windows


class WashDataUI:
    def __init__(self, mode: str = 'cleaning', version: str = '1'):
        self.mode = mode
        self.version = version
        self.signal_names = self._get_signal_names()
        self.fig, self.axes = None, {}
        self.sliders: dict[str, Slider] = {}
        self.callbacks: dict[str, callable] = {}
        self._init_plot()

    def _get_signal_names(self):
        if self.version == '2':
            return ['ecg', 'rppg', 'ppg_red', 'ppg_ir', 'ppg_green']
        return ['ecg', 'rppg']

    def _init_plot(self):
        n_signals = len(self.signal_names)
        self.fig, axes_list = plt.subplots(n_signals, 1, figsize=(16, 3 * n_signals))
        if n_signals == 1:
            axes_list = [axes_list]
        for name, axis in zip(self.signal_names, axes_list):
            self.axes[name] = axis
        plt.subplots_adjust(bottom=0.2 if self.mode == 'cleaning' else 0.1)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        if self.mode == 'cleaning':
            self._init_sliders()
            self.fig.suptitle("Controls: 'y' Accept, 'n' Reject, 'q' Quit", fontsize=12)
        else:
            self.fig.suptitle("Controls: 'y' Accept, 'n' Reject, 'r' Reverse, 'q' Quit", fontsize=12)

    def _init_sliders(self):
        if self.version == '2':
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.05, 0.02, 0.12, 0.03),
                ('rppg_std', 'RPPG STD', 0.20, 0.02, 0.12, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.35, 0.02, 0.12, 0.03),
                ('ppg_red_std', 'Red STD', 0.50, 0.02, 0.12, 0.03),
                ('ppg_ir_std', 'IR STD', 0.65, 0.02, 0.12, 0.03),
                ('ppg_green_std', 'Green STD', 0.80, 0.02, 0.12, 0.03),
            ]
        else:
            slider_configs = [
                ('ecg_std', 'ECG STD', 0.1, 0.05, 0.2, 0.03),
                ('rppg_std', 'RPPG STD', 0.4, 0.05, 0.2, 0.03),
                ('rppg_bpm', 'RPPG BPM', 0.7, 0.05, 0.2, 0.03),
            ]

        for name, label, left, bottom, width, height in slider_configs:
            ax = plt.axes([left, bottom, width, height])
            vmin, vmax, vinit, step = (0.5, 3.0, 1.5, 0.1) if 'std' in name else (5, 30, 15, 1)
            slider = Slider(ax, label, vmin, vmax, valinit=vinit, valstep=step)
            slider.on_changed(lambda val, n=name: self._on_slider_change(n))
            self.sliders[name] = slider

    def update_plot(self, data: SignalData, masks: dict[str, np.ndarray] | None = None, title_suffix: str = ""):
        for name in self.signal_names:
            axis = self.axes[name]
            axis.clear()
            values = data.get_signal(name)
            if values.size == 0:
                continue
            axis.plot(data.time, values, label=name.upper())
            if masks and name in masks and masks[name].size:
                axis.fill_between(data.time, values, where=~masks[name], color='red', alpha=0.3, label='Artifact')
            axis.set_title(f'{name.upper()} Signal {title_suffix}')
            axis.grid(True, alpha=0.3)
            axis.legend(loc='upper right')
        self.fig.canvas.draw_idle()

    def get_config(self):
        config = {
            'ecg': {'std': {'window_size': 1, 'threshold': 1.5}},
            'rppg': {
                'std': {'window_size': 1, 'threshold': 1.5},
                'welch': {'window_size': 5, 'bpm_tolerance': 15},
            },
        }
        if self.version == '2':
            for sig in ['ppg_red', 'ppg_ir', 'ppg_green']:
                config[sig] = {'std': {'window_size': 1, 'threshold': 1.5}}

        for name, slider in self.sliders.items():
            val = slider.val
            if name == 'ecg_std':
                config['ecg']['std']['threshold'] = val
            elif name == 'rppg_std':
                config['rppg']['std']['threshold'] = val
            elif name == 'rppg_bpm':
                config['rppg']['welch']['bpm_tolerance'] = val
            elif name == 'ppg_red_std' and 'ppg_red' in config:
                config['ppg_red']['std']['threshold'] = val
            elif name == 'ppg_ir_std' and 'ppg_ir' in config:
                config['ppg_ir']['std']['threshold'] = val
            elif name == 'ppg_green_std' and 'ppg_green' in config:
                config['ppg_green']['std']['threshold'] = val
        return config

    def on_action(self, callback):
        self.callbacks['action'] = callback

    def on_config_change(self, callback):
        self.callbacks['config'] = callback

    def _on_key(self, event):
        if event.key in ['y', 'n', 'r', 'q'] and 'action' in self.callbacks:
            self.callbacks['action'](event.key)

    def _on_slider_change(self, name):
        if 'config' in self.callbacks:
            self.callbacks['config']()


class WashDataController:
    def __init__(self, data_path: str, log_path: str, start: int = 0, end: int | None = None):
        self.data_path = Path(data_path)
        self.log_path = Path(log_path)
        self.start = start
        self.end = end
        self.version = global_vars.mirror_version
        self.loader = DataLoader(self.version)
        self.cleaner = SignalCleaner()
        self.logger = DataLogger(log_path)
        self.ui: WashDataUI | None = None
        self.files: list[str] = []
        self.current_idx = 0
        self.current_data: SignalData | None = None

    def run_cleaning(self):
        self.files = self._collect_files(self.data_path, raw=True)
        if not self.files:
            print('No files to process.')
            return
        self.ui = WashDataUI(mode='cleaning', version=self.version)
        self.ui.on_action(self._handle_cleaning_action)
        self.ui.on_config_change(self._update_cleaning_view)
        self._load_next_raw()
        plt.show(block=True)

    def run_checking(self):
        self.files = self._collect_files(self.log_path, raw=False)
        if not self.files:
            print('No cleaned files to check.')
            return
        self.ui = WashDataUI(mode='checking', version=self.version)
        self.ui.on_action(self._handle_checking_action)
        self._load_next_cleaned()
        plt.show(block=True)

    def _collect_files(self, location: Path, raw: bool) -> list[str]:
        if not location.exists():
            return []
        entries = []
        for entry in sorted(os.listdir(location)):
            full_path = location / entry
            if raw and not full_path.is_dir():
                continue
            if not raw and not entry.endswith('.csv'):
                continue
            identifier = entry.split('_')[-1] if raw else entry.split('_')[1]
            try:
                pid = int(identifier.rstrip('.csv')) if raw else int(identifier)
            except ValueError:
                continue
            if pid < self.start or (self.end is not None and pid > self.end):
                continue
            entries.append(entry)
        return entries

    def _load_next_raw(self):
        if self.current_idx >= len(self.files):
            print('All files processed.')
            plt.close('all')
            return
        folder_name = self.files[self.current_idx]
        print(f"Processing {folder_name} ({self.current_idx + 1}/{len(self.files)})...")
        self.current_data = self.loader.load(str(self.data_path / folder_name))
        self._update_cleaning_view()

    def _update_cleaning_view(self):
        if not self.ui or self.current_data is None:
            return
        config = self.ui.get_config()
        masks = {}
        for name in self.ui.signal_names:
            if name in config:
                masks[name] = self.cleaner.clean(self.current_data.get_signal(name), config[name])
        self.ui.update_plot(self.current_data, masks, title_suffix=f"- {self.files[self.current_idx]}")

    def _handle_cleaning_action(self, action):
        if action == 'q':
            plt.close('all')
            return
        if action == 'y' and self.current_data is not None:
            config = self.ui.get_config()
            masks = {}
            for name in self.ui.signal_names:
                if name in config:
                    masks[name] = self.cleaner.clean(self.current_data.get_signal(name), config[name])
            self.logger.log_cleaned_data(f'{self.files[self.current_idx]}.csv', self.current_data, masks)
        self.current_idx += 1
        self._load_next_raw()

    def _load_next_cleaned(self):
        if self.current_idx >= len(self.files):
            print('All files checked.')
            plt.close('all')
            return
        file_name = self.files[self.current_idx]
        print(f"Checking {file_name} ({self.current_idx + 1}/{len(self.files)})...")
        df = pd.read_csv(self.log_path / file_name)
        df = _standardize_columns(df)
        for col in df.columns:
            if col == 'Time':
                continue
            df[col] = (df[col] - df[col].mean()) / (df[col].std(ddof=0) or 1)
        self.current_data = SignalData(df)
        self.ui.update_plot(self.current_data, title_suffix=f"- {file_name}")

    def _handle_checking_action(self, action):
        if action == 'q':
            plt.close('all')
            return
        file_name = self.files[self.current_idx]
        if action == 'n':
            self.logger.modify_cleaned_data(file_name, 'reject')
        elif action == 'r':
            self.logger.modify_cleaned_data(file_name, 'reverse')
        self.current_idx += 1
        self._load_next_cleaned()


def main():
    data_path = input('Input data path:').strip()
    log_path = input('Input log path:').strip()
    start = input('Input starting point (default 0):').strip()
    end = input('Input ending point (default None):').strip()

    start_idx = int(start) if start.isdigit() else 0
    end_idx = int(end) if end.isdigit() else None

    controller = WashDataController(data_path, log_path, start_idx, end_idx)

    print('\n--- Starting Cleaning Phase ---')
    controller.run_cleaning()

    controller.current_idx = 0
    print('\n--- Starting Checking Phase ---')
    controller.run_checking()


if __name__ == '__main__':
    main()
