import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SBP/DBP distributions across merged patient info CSV files."
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=".",
        help="Directory containing merged_patient_info_*.csv files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "merged_patient_info_1.csv",
            "merged_patient_info_2.csv",
            "merged_patient_info_4.csv",
            "merged_patient_info_5.csv",
            "merged_patient_info_6.csv",
        ],
        help="CSV file names to include in the visualization.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="bp_distribution_all_merged.png",
        help="Output image path. If empty, the figure is not saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="If set, do not display the plot window.",
    )
    return parser.parse_args()


def _find_column(df: pd.DataFrame, candidate_names: list[str]) -> str:
    normalized = {col.strip().lower(): col for col in df.columns}
    for name in candidate_names:
        col = normalized.get(name.lower())
        if col is not None:
            return col
    raise ValueError(f"Missing required columns. Tried: {candidate_names}")


def load_bp_data(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    dbp_col = _find_column(df, ["low_blood_pressure", "Low_Blood_Pressure"])
    sbp_col = _find_column(df, ["high_blood_pressure", "High_Blood_Pressure"])

    dbp = pd.to_numeric(df[dbp_col], errors="coerce")
    sbp = pd.to_numeric(df[sbp_col], errors="coerce")

    # Filter out sentinel/missing values and implausible extremes.
    dbp = dbp[(dbp > 0) & (dbp <= 200)]
    sbp = sbp[(sbp > 0) & (sbp <= 300)]

    return dbp, sbp


def plot_distributions(series_by_file: dict[str, tuple[pd.Series, pd.Series]], bins: int) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    color_cycle = plt.cm.tab10.colors

    for idx, (label, (dbp, sbp)) in enumerate(series_by_file.items()):
        color = color_cycle[idx % len(color_cycle)]
        axes[0].hist(
            dbp,
            bins=bins,
            alpha=0.35,
            histtype="stepfilled",
            edgecolor=color,
            facecolor=color,
            linewidth=1.2,
            label=f"{label} (n={len(dbp)})",
        )
        axes[1].hist(
            sbp,
            bins=bins,
            alpha=0.35,
            histtype="stepfilled",
            edgecolor=color,
            facecolor=color,
            linewidth=1.2,
            label=f"{label} (n={len(sbp)})",
        )

    axes[0].set_title("DBP Distribution (Low Blood Pressure)")
    axes[0].set_xlabel("Diastolic BP (mmHg)")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("SBP Distribution (High Blood Pressure)")
    axes[1].set_xlabel("Systolic BP (mmHg)")
    axes[1].grid(alpha=0.25)

    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    fig.suptitle("Blood Pressure Distribution Across merged_patient_info_*.csv", fontsize=13)
    fig.tight_layout()
    return fig


def _format_stat(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.2f}"


def print_bp_spread_stats(series_by_file: dict[str, tuple[pd.Series, pd.Series]]) -> None:
    print("\nBP spread summary per CSV (SBP/DBP):")
    print("-" * 86)
    print(
        "file".ljust(28)
        + "DBP std(sample)".rjust(15)
        + "DBP stddev(pop)".rjust(17)
        + "SBP std(sample)".rjust(15)
        + "SBP stddev(pop)".rjust(17)
    )
    print("-" * 86)

    all_dbp = []
    all_sbp = []

    for label, (dbp, sbp) in series_by_file.items():
        dbp_sample_std = dbp.std(ddof=1)
        dbp_pop_stddev = dbp.std(ddof=0)
        sbp_sample_std = sbp.std(ddof=1)
        sbp_pop_stddev = sbp.std(ddof=0)

        print(
            label.ljust(28)
            + _format_stat(dbp_sample_std).rjust(15)
            + _format_stat(dbp_pop_stddev).rjust(17)
            + _format_stat(sbp_sample_std).rjust(15)
            + _format_stat(sbp_pop_stddev).rjust(17)
        )

        all_dbp.append(dbp)
        all_sbp.append(sbp)

    merged_dbp = pd.concat(all_dbp, ignore_index=True)
    merged_sbp = pd.concat(all_sbp, ignore_index=True)

    total_dbp_sample_std = merged_dbp.std(ddof=1)
    total_dbp_pop_stddev = merged_dbp.std(ddof=0)
    total_sbp_sample_std = merged_sbp.std(ddof=1)
    total_sbp_pop_stddev = merged_sbp.std(ddof=0)

    print("-" * 86)
    print(
        "TOTAL".ljust(28)
        + _format_stat(total_dbp_sample_std).rjust(15)
        + _format_stat(total_dbp_pop_stddev).rjust(17)
        + _format_stat(total_sbp_sample_std).rjust(15)
        + _format_stat(total_sbp_pop_stddev).rjust(17)
    )
    print("-" * 86)


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)

    series_by_file: dict[str, tuple[pd.Series, pd.Series]] = {}

    for file_name in args.files:
        csv_path = csv_dir / file_name
        if not csv_path.exists():
            print(f"Warning: file not found, skipping: {csv_path}")
            continue

        try:
            dbp, sbp = load_bp_data(csv_path)
        except Exception as exc:
            print(f"Warning: failed to load {csv_path}: {exc}")
            continue

        label = csv_path.stem
        series_by_file[label] = (dbp, sbp)
        print(f"Loaded {label}: DBP n={len(dbp)}, SBP n={len(sbp)}")

    if not series_by_file:
        raise RuntimeError("No valid CSV data found to plot.")

    print_bp_spread_stats(series_by_file)

    fig = plot_distributions(series_by_file, bins=args.bins)

    if args.save:
        output_path = Path(args.save)
        fig.savefig(output_path, dpi=200)
        print(f"Saved plot to: {output_path.resolve()}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
