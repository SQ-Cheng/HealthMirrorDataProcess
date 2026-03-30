import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATTERNS = ["mirror*_auto_cleaned_sqi"]


@dataclass
class ViolationRecord:
    csv_path: Path
    folder_type: str
    mirror_id: str
    max_val: float
    min_val: float


def detect_ecg_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "ecg":
            return col
    raise ValueError("ECG column not found")


def find_violations(root: Path, patterns: List[str]) -> List[ViolationRecord]:
    violations: List[ViolationRecord] = []

    for pattern in patterns:
        for folder in sorted(root.glob(pattern)):
            if not folder.is_dir():
                continue

            folder_type = "cleaned_sqi" if folder.name.endswith("_sqi") else "cleaned"
            match = re.search(r"mirror(\d+)", folder.name, flags=re.IGNORECASE)
            mirror_id = match.group(1) if match else "unknown"

            for csv_path in sorted(folder.glob("*.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    ecg_col = detect_ecg_column(df)
                    ecg = pd.to_numeric(df[ecg_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
                    if len(ecg) == 0:
                        continue

                    max_val = float(np.max(ecg))
                    min_val = float(np.min(ecg))

                    # Rule: max should be strictly greater than abs(min)
                    if max_val <= abs(min_val):
                        violations.append(
                            ViolationRecord(
                                csv_path=csv_path,
                                folder_type=folder_type,
                                mirror_id=mirror_id,
                                max_val=max_val,
                                min_val=min_val,
                            )
                        )
                except Exception as e:
                    print(f"[Warn] Skipping {csv_path}: {e}")

    return violations


class ECGPolarityReviewer:
    def __init__(self, violations: List[ViolationRecord], apply_on_exit: bool = False):
        self.violations = violations
        self.idx = 0
        # Default decision for violations: flip
        self.flip_decision = {v.csv_path: True for v in violations}
        self.apply_on_exit = apply_on_exit
        self.applied = False

        self.fig, self.ax = plt.subplots(figsize=(13, 5))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _load_signal(self, path: Path):
        df = pd.read_csv(path)
        ecg_col = detect_ecg_column(df)
        ecg = pd.to_numeric(df[ecg_col], errors="coerce").to_numpy(dtype=np.float64)
        return df, ecg_col, ecg

    def redraw(self):
        rec = self.violations[self.idx]
        df, ecg_col, ecg = self._load_signal(rec.csv_path)
        decision_flip = self.flip_decision[rec.csv_path]

        self.ax.clear()
        self.ax.plot(ecg, color="black", linewidth=0.8, alpha=0.8, label="ECG current")

        if decision_flip:
            self.ax.plot(-ecg, color="tab:orange", linewidth=0.8, alpha=0.8, label="ECG after flip preview")

        max_val = float(np.nanmax(ecg))
        min_val = float(np.nanmin(ecg))
        rule_ok_now = max_val > abs(min_val)

        title = (
            f"[{self.idx + 1}/{len(self.violations)}] mirror{rec.mirror_id} | {rec.csv_path.name} | {rec.folder_type} | "
            f"max={max_val:.4f}, min={min_val:.4f}, abs(min)={abs(min_val):.4f} | "
            f"rule_now={'PASS' if rule_ok_now else 'FAIL'} | decision={'FLIP' if decision_flip else 'KEEP'}"
        )
        self.ax.set_title(title)
        self.ax.set_xlabel("Sample index")
        self.ax.set_ylabel(ecg_col)
        self.ax.grid(alpha=0.2)
        self.ax.legend(loc="upper right")

        help_text = (
            "Left/Right: prev/next | F or Space: toggle flip/keep | A: apply flips + save log | "
            "L: save decision log only | Q: quit"
        )
        self.fig.text(0.01, 0.01, help_text, fontsize=9)
        self.fig.tight_layout(rect=[0, 0.03, 1, 1])
        self.fig.canvas.draw_idle()

    def _decision_table(self):
        rows = []
        for rec in self.violations:
            rows.append(
                {
                    "csv_path": str(rec.csv_path),
                    "folder_type": rec.folder_type,
                    "mirror_id": rec.mirror_id,
                    "max_before": rec.max_val,
                    "min_before": rec.min_val,
                    "abs_min_before": abs(rec.min_val),
                    "decision_flip": self.flip_decision[rec.csv_path],
                }
            )
        return pd.DataFrame(rows)

    def save_log(self):
        out = self._decision_table()
        out_path = Path.cwd() / "ecg_flip_decisions.csv"
        out.to_csv(out_path, index=False)
        print(f"[Info] Decision log saved to: {out_path}")

    def apply_flips(self):
        flip_count = 0
        for rec in self.violations:
            if not self.flip_decision[rec.csv_path]:
                continue

            df = pd.read_csv(rec.csv_path)
            ecg_col = detect_ecg_column(df)
            df[ecg_col] = -pd.to_numeric(df[ecg_col], errors="coerce")
            df.to_csv(rec.csv_path, index=False)
            flip_count += 1

        self.applied = True
        print(f"[Info] Applied ECG polarity flip to {flip_count} file(s).")
        self.save_log()

    def on_key(self, event):
        key = (event.key or "").lower()

        if key == "right":
            self.idx = min(self.idx + 1, len(self.violations) - 1)
            self.redraw()
        elif key == "left":
            self.idx = max(self.idx - 1, 0)
            self.redraw()
        elif key in {"f", " "}:
            rec = self.violations[self.idx]
            self.flip_decision[rec.csv_path] = not self.flip_decision[rec.csv_path]
            self.redraw()
        elif key == "l":
            self.save_log()
        elif key == "a":
            self.apply_flips()
            plt.close(self.fig)
        elif key == "q":
            if self.apply_on_exit and not self.applied:
                self.apply_flips()
            else:
                self.save_log()
            plt.close(self.fig)

    def run(self):
        if not self.violations:
            print("No violating files found. Nothing to review.")
            return

        print(f"Found {len(self.violations)} violating file(s).")
        print("Interactive keys: Left/Right to navigate, F/Space to toggle flip, A apply+save, L log, Q quit")
        self.redraw()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Review ECG polarity in cleaned_sqi files where max <= abs(min)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Workspace root to search for mirror*_auto_cleaned_sqi folders",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="Folder glob patterns to scan",
    )
    parser.add_argument(
        "--apply_on_exit",
        action="store_true",
        help="If set, pressing Q also applies selected flips before closing",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations = find_violations(root, args.patterns)

    if violations:
        mirror_counts = Counter(v.mirror_id for v in violations)
        print("\nViolation statistics (max <= abs(min)):")
        for mirror_id in sorted(mirror_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            print(f"mirror{mirror_id}: {mirror_counts[mirror_id]} file(s)")
        print(f"TOTAL: {len(violations)} file(s)")
    else:
        print("No violating files found.")

    reviewer = ECGPolarityReviewer(violations, apply_on_exit=args.apply_on_exit)
    reviewer.run()


if __name__ == "__main__":
    main()
