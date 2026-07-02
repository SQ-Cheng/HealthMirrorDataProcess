"""
Merge 7 lab test CSV files into one combined CSV.

Files:
  - 乳酸.csv        (Lactate)
  - 二氧化碳分压.csv  (CO2 Partial Pressure)
  - 心肌酶.csv       (Cardiac Enzymes - hsTnI)
  - 氧分压.csv       (O2 Partial Pressure / Blood Gas)
  - 葡萄糖浓度.csv    (Glucose Concentration)
  - 血糖.csv         (Blood Glucose)
  - 血色素.csv       (Hemoglobin)

Each file has: Row1=description, Row2=headers (18+trailing comma->19 cols), Rows3+=data
Corrupted rows (from 心肌酶.csv where data shifted) are automatically filtered out.
"""

import pandas as pd
import os

CSV_FILES = [
    "乳酸.csv",
    "二氧化碳分压.csv",
    "心肌酶.csv",
    "氧分压.csv",
    "葡萄糖浓度.csv",
    "血糖.csv",
    "血色素.csv",
]

OUTPUT_FILE = "merged_lab_tests.csv"

BASE_DIR = "/root/autodl-tmp/HealthMirrorDataProcess"

# Known valid test item names containing Chinese/lab characters
# Used to filter corrupted rows (pure numbers, gender chars, dates etc)


def is_valid_test_item(name: str) -> bool:
    """Check if a test item name looks like a valid lab test (contains Chinese chars)."""
    if not name or not name.strip():
        return False
    # Must contain at least one Chinese character or common lab test symbols
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in name)
    if not has_chinese:
        return False
    # Exclude rows that are clearly just demographic info (e.g. "男", "女")
    if name.strip() in ('男', '女'):
        return False
    # Exclude rows that look like patient IDs (long numbers)
    if name.strip().isdigit() and len(name.strip()) >= 4:
        return False
    return True


def read_lab_csv(filepath: str) -> pd.DataFrame:
    """Read a lab CSV file, skipping the description row and handling trailing comma."""
    df = pd.read_csv(
        filepath,
        encoding="gbk",
        skiprows=1,
        header=0,
        dtype=str,
        keep_default_na=False,
    )
    # Drop trailing unnamed empty columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    df.columns = df.columns.str.strip()
    return df


def clean_corrupted_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where test item name is corrupted/not a real lab test."""
    before = len(df)
    mask = df["检验项名称"].apply(is_valid_test_item)
    df = df[mask].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"  Removed {removed} corrupted rows")
    return df


def validate_row_counts(files_data: dict) -> None:
    """Print row counts per file and total."""
    total = 0
    print("=" * 60)
    print(f"{'File':<25s} {'Rows':>8s}")
    print("-" * 60)
    for fname, df in files_data.items():
        n = len(df)
        total += n
        print(f"{fname:<25s} {n:>8d}")
    print("-" * 60)
    print(f"{'TOTAL (raw)':<25s} {total:>8d}")
    print("=" * 60)


def main():
    os.chdir(BASE_DIR)

    files_data = {}
    for fname in CSV_FILES:
        print(f"Reading {fname}...")
        df = read_lab_csv(fname)
        if len(df.columns) != 18:
            print(f"  WARNING: {fname} has {len(df.columns)} columns, expected 18")
        files_data[fname] = df

    validate_row_counts(files_data)

    # Concatenate
    merged = pd.concat(files_data.values(), ignore_index=True)
    print(f"\nRaw merged: {len(merged)} rows x {len(merged.columns)} columns")

    # Clean corrupted rows
    merged = clean_corrupted_rows(merged)
    print(f"Clean merged: {len(merged)} rows x {len(merged.columns)} columns")

    # Save
    merged.to_csv(OUTPUT_FILE, index=False, encoding="gbk")
    print(f"\nSaved to {OUTPUT_FILE}")

    # Verification
    verify = pd.read_csv(OUTPUT_FILE, encoding="gbk", dtype=str, keep_default_na=False)
    assert len(verify) == len(merged), f"Row count mismatch: {len(verify)} != {len(merged)}"
    print("Verification PASSED!")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Total rows: {len(verify)}")
    print(f"Unique patients: {verify['首页病案号'].nunique()}")
    print(f"\nTest items ({verify['检验项名称'].nunique()} unique):")
    for item, count in verify["检验项名称"].value_counts().items():
        print(f"  {item}: {count}")


if __name__ == "__main__":
    main()
