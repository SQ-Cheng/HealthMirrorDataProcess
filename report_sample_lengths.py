from pathlib import Path
import csv


def count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def folder_total_rows(folder: Path) -> int:
    total = 0
    for csv_file in sorted(folder.glob("*.csv")):
        total += count_csv_rows(csv_file)
    return total


root = Path(__file__).resolve().parent

cleaned_dirs = sorted(root.glob("mirror*_auto_cleaned"))
sqi_dirs = sorted(root.glob("mirror*_auto_cleaned_sqi"))
typo_dirs = sorted(root.glob("mirror*_auto_claened_sqi"))

print("=== mirrorx_auto_cleaned totals ===")
if not cleaned_dirs:
    print("(none found)")
cleaned_grand_total = 0
for d in cleaned_dirs:
    total = folder_total_rows(d)
    cleaned_grand_total += total
    print(f"{d.name}: {total}")
print(f"GROUP_TOTAL(mirror*_auto_cleaned): {cleaned_grand_total}")

print("\n=== mirrorx_auto_cleaned_sqi totals ===")
if not sqi_dirs:
    print("(none found)")
sqi_grand_total = 0
for d in sqi_dirs:
    total = folder_total_rows(d)
    sqi_grand_total += total
    print(f"{d.name}: {total}")
print(f"GROUP_TOTAL(mirror*_auto_cleaned_sqi): {sqi_grand_total}")
