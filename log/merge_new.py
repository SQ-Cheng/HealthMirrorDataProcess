import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import global_vars

class FileMerger:
    def __init__(self, path=None, log=False, resample_rate=512) -> None:
        self.path = path
        self.log = log
        self.resample_rate = resample_rate
        self.raw_data = dict()
        self.data = dict()
        self.mirror_version = global_vars.mirror_version

    def _load_data(self) -> None:
        def _load_data_v1() -> dict:
            rppg_df = pd.read_csv(os.path.join(self.path, "rppg_log.csv"), dtype=float)
            ecg_df = pd.read_csv(os.path.join(self.path, "ecg_log.csv"), dtype=float)
            # TODO: change to column names
            rppg_timestamp = rppg_df["timestamp"].to_numpy().tolist()
            rppg_signal = rppg_df["rppg"].to_numpy().tolist()
            ecg_timestamp = ecg_df.iloc[:, 0].to_numpy().tolist()
            ecg_signal = ecg_df.iloc[:, 1].to_numpy().tolist()
            return {
                "RPPG_Timestamp": rppg_timestamp,
                "RPPG_Signal": rppg_signal,
                "ECG_Timestamp": ecg_timestamp,
                "ECG_Signal": ecg_signal,
                "First_Timestamp": min(rppg_timestamp[0], ecg_timestamp[0]),
                "Last_Timestamp": max(rppg_timestamp[-1], ecg_timestamp[-1])
            }

        def _load_data_v2() -> dict:
            rppg_df = pd.read_csv(os.path.join(self.path, "rppg_log.csv"), dtype=float)
            ppg_df = pd.read_csv(os.path.join(self.path, "ppg_log.csv"), dtype=float)
            ecg_df = pd.read_csv(os.path.join(self.path, "ecg_log.csv"), dtype=float)
            rppg_timestamp = rppg_df["timestamp"].to_numpy().tolist()
            rppg_signal = rppg_df["rppg"].to_numpy().tolist()
            ppg_timestamp = ppg_df["timestamp"].to_numpy().tolist()
            ppg_red = ppg_df["ppg_red"].to_numpy().tolist()
            ppg_ir = ppg_df["ppg_ir"].to_numpy().tolist()
            ppg_green = ppg_df["ppg_green"].to_numpy().tolist()
            ecg_timestamp = ecg_df["timestamp"].to_numpy().tolist()
            ecg_signal = ecg_df["ecg"].to_numpy().tolist()
            return {
                "RPPG_Timestamp": rppg_timestamp,
                "RPPG_Signal": rppg_signal,
                "PPG_Timestamp": ppg_timestamp,
                "PPG_Red": ppg_red,
                "PPG_IR": ppg_ir,
                "PPG_Green": ppg_green,
                "ECG_Timestamp": ecg_timestamp,
                "ECG_Signal": ecg_signal,
                "First_Timestamp": min(rppg_timestamp[0], ppg_timestamp[0], ecg_timestamp[0]),
                "Last_Timestamp": max(rppg_timestamp[-1], ppg_timestamp[-1], ecg_timestamp[-1])
            }
        
        if self.log:
            print(f"[FileMerger] Loading data from {self.path}...")
        if self.mirror_version == "1":
            self.raw_data = _load_data_v1()
        elif self.mirror_version == "2":
            self.raw_data = _load_data_v2()
        if self.log:
            print(f"[FileMerger] Data loaded.")
    
    def _resample_data(self):
        def resample_signal(original_timestamps, original_signal, new_timestamps):
            return interp1d(original_timestamps, original_signal, kind='cubic', fill_value=0, bounds_error=False)(new_timestamps)

        if self.log:
            print(f"[FileMerger] Resampling data to {self.resample_rate} Hz...")
        new_timestamps = np.linspace(
            self.raw_data["First_Timestamp"],
            self.raw_data["Last_Timestamp"],
            int((self.raw_data["Last_Timestamp"] - self.raw_data["First_Timestamp"]) * self.resample_rate)
        )

        self.data["Timestamp"] = new_timestamps.tolist()
        self.data["RPPG"] = resample_signal(
            self.raw_data["RPPG_Timestamp"], self.raw_data["RPPG_Signal"], new_timestamps
        ).tolist()
        self.data["ECG"] = resample_signal(
            self.raw_data["ECG_Timestamp"], self.raw_data["ECG_Signal"], new_timestamps
        ).tolist()

        if self.mirror_version == "2":
            self.data["PPG_RED"] = resample_signal(
                self.raw_data["PPG_Timestamp"], self.raw_data["PPG_Red"], new_timestamps
            ).tolist()
            self.data["PPG_IR"] = resample_signal(
                self.raw_data["PPG_Timestamp"], self.raw_data["PPG_IR"], new_timestamps
            ).tolist()
            self.data["PPG_GREEN"] = resample_signal(
                self.raw_data["PPG_Timestamp"], self.raw_data["PPG_Green"], new_timestamps
            ).tolist()

        if self.log:
            print(f"[FileMerger] Resampling complete.")

    def __call__(self):
        self._load_data()
        self._resample_data()
        self.merged_df = pd.DataFrame(self.data)
        self.merged_df.to_csv(os.path.join(self.path, "merged_log.csv"), index=False)
        if self.log:
            print(f"[FileMerger] Merged data saved to {os.path.join(self.path, 'merged_log.csv')}")
        return self.merged_df
        
        
    



        
        
        