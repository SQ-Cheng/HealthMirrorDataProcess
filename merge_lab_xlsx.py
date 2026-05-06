"""
Merge lab test data from 健康镜化验1.0.xlsx into existing merged_patient_info CSV files.

The XLSX contains one-to-many rows (one patient, multiple lab tests over time).
This script aggregates the lab data per patient and left-joins onto the CSV
using hospital_patient_id <-> 首页病案号.

Usage:
    python merge_lab_xlsx.py                          # default: mirror 6
    python merge_lab_xlsx.py --mirror-id 1
    python merge_lab_xlsx.py --csv merged_patient_info_6.csv --xlsx 健康镜化验1.0.xlsx --output merged_with_lab.csv
"""

import argparse
import numpy as np
import pandas as pd


def parse_xlsx(xlsx_path):
    """
    Parse the lab XLSX file with merged headers and aggregate per patient.

    Returns:
        DataFrame with one row per patient, indexed by hospital_patient_id (int).
        Columns: gender, age, lactate_min, lactate_max, lactate_mean,
                 lactate_median, lactate_std, lactate_count,
                 lactate_first_report_time, lactate_last_report_time
    """
    df = pd.read_excel(xlsx_path, header=[0, 1])

    # Flatten multi-index columns to level-1 names
    df.columns = [col[1] if col[1] else col[0] for col in df.columns]

    # Convert patient ID to int
    df["hospital_patient_id"] = pd.to_numeric(df["首页病案号"], errors="coerce")
    df = df.dropna(subset=["hospital_patient_id"])
    df["hospital_patient_id"] = df["hospital_patient_id"].astype(int)

    # Parse lactate values to float
    df["lactate_value"] = pd.to_numeric(df["检验值(文本)"], errors="coerce")

    # Parse report time
    df["report_time"] = pd.to_datetime(df["报告时间"], errors="coerce")

    # Extract age as integer from "XX岁"
    df["age"] = df["首页就诊时年龄"].str.replace("岁", "", regex=False)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Aggregate per patient
    agg_df = df.groupby("hospital_patient_id").agg(
        gender=("首页性别", "first"),
        age=("age", "first"),
        lactate_min=("lactate_value", "min"),
        lactate_max=("lactate_value", "max"),
        lactate_mean=("lactate_value", "mean"),
        lactate_median=("lactate_value", "median"),
        lactate_std=("lactate_value", "std"),
        lactate_count=("lactate_value", "count"),
        lactate_first_report_time=("report_time", "min"),
        lactate_last_report_time=("report_time", "max"),
    ).reset_index()

    # Round float columns to 2 decimal places
    float_cols = ["lactate_min", "lactate_max", "lactate_mean", "lactate_median", "lactate_std"]
    for col in float_cols:
        agg_df[col] = agg_df[col].round(2)

    # Format report times as strings for CSV output
    for col in ["lactate_first_report_time", "lactate_last_report_time"]:
        agg_df[col] = agg_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        agg_df[col] = agg_df[col].fillna("-1")

    return agg_df


def merge_with_csv(csv_path, lab_df, output_path):
    """
    Left-join lab data onto the existing merged patient info CSV.

    Args:
        csv_path: Path to existing merged_patient_info CSV
        lab_df: Aggregated lab DataFrame from parse_xlsx()
        output_path: Path to write output CSV
    """
    df_csv = pd.read_csv(csv_path)

    # Ensure hospital_patient_id is int for matching
    df_csv["hospital_patient_id"] = pd.to_numeric(df_csv["hospital_patient_id"], errors="coerce")

    # Left join
    df_merged = df_csv.merge(lab_df, on="hospital_patient_id", how="left")

    # Fill missing lab columns with -1 (for numeric) or -1 (for string)
    lab_numeric_cols = ["age", "lactate_min", "lactate_max", "lactate_mean",
                        "lactate_median", "lactate_std", "lactate_count"]
    lab_string_cols = ["gender", "lactate_first_report_time", "lactate_last_report_time"]

    for col in lab_numeric_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(-1)

    for col in lab_string_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna("-1")

    # Reorder columns: original CSV columns first, then new lab columns
    original_cols = list(df_csv.columns)
    new_cols = [c for c in df_merged.columns if c not in original_cols]
    df_merged = df_merged[original_cols + new_cols]

    df_merged.to_csv(output_path, index=False)

    # Print summary
    matched = df_merged[df_merged["lactate_count"] != -1]
    print(f"=== Merge Summary ===")
    print(f"Source CSV:  {csv_path} ({len(df_csv)} rows)")
    print(f"Lab XLSX:    {len(lab_df)} unique patients")
    print(f"Matched:     {len(matched)} patients have lab data")
    print(f"Unmatched:   {len(df_csv) - len(matched)} patients have no lab data")
    print(f"New columns: {new_cols}")
    print(f"Output:      {output_path}")

    # Show sample of matched rows
    if len(matched) > 0:
        print(f"\n=== Sample matched rows (first 5) ===")
        display_cols = ["lab_patient_id", "hospital_patient_id", "gender", "age",
                        "lactate_min", "lactate_max", "lactate_mean", "lactate_count"]
        display_cols = [c for c in display_cols if c in df_merged.columns]
        print(matched[display_cols].head().to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge lab test data from XLSX into merged_patient_info CSV"
    )
    parser.add_argument("--mirror-id", type=int, default=6,
                        help="Mirror ID for file naming (default: 6)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to existing merged_patient_info CSV")
    parser.add_argument("--xlsx", type=str, default="健康镜化验1.0.xlsx",
                        help="Path to lab XLSX file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV")
    args = parser.parse_args(argv)

    csv_path = args.csv or f"merged_patient_info_{args.mirror_id}.csv"
    output_path = args.output or f"merged_patient_info_with_lab_{args.mirror_id}.csv"

    print(f"Parsing lab data from {args.xlsx}...")
    lab_df = parse_xlsx(args.xlsx)
    print(f"Found {len(lab_df)} patients with lab data\n")

    merge_with_csv(csv_path, lab_df, output_path)


if __name__ == "__main__":
    main()