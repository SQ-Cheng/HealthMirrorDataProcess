"""
Scan all mirrorx_data/patient_xxxxxx/patient_info.txt files.
Identify electrical-stimulation patients: two sessions with the same
hospital_patient_id (compared as int) whose session timestamps fall on
the same calendar day.  Output their patient_id, hospital_patient_id,
and mirror_id to a CSV.
"""

import os
import re
import csv
import json
import glob
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "electrical_stimulation_patients.csv")

MIRROR_PATTERN = os.path.join(BASE_DIR, "mirror*_data")
PATIENT_FOLDER_RE = re.compile(r"^patient_\d+$")
PATIENT_ID_RE = re.compile(r"^Patient ID:\s*(\S+)", re.MULTILINE)
TIMESTAMP_RE = re.compile(r"^Session Timestamp:\s*(.+)$", re.MULTILINE)
MIRROR_ID_RE = re.compile(r"mirror(\d+)_data")


def extract_hospital_patient_id(text: str) -> str:
    """Extract patient_id (hospital_patient_id) from the Patient Info JSON."""
    match = re.search(r'Patient Info:\s*"?(.*)"?\s*$', text, re.MULTILINE)
    if not match:
        return ""
    raw = match.group(1).strip().strip('"')
    raw = raw.replace('\\"', '"')
    try:
        info = json.loads(raw)
        return info.get("patient_id", "")
    except (json.JSONDecodeError, TypeError):
        return ""


def extract_session_date(text: str) -> str:
    """Extract the date portion (YYYY-MM-DD) from Session Timestamp."""
    match = TIMESTAMP_RE.search(text)
    if not match:
        return ""
    ts_str = match.group(1).strip()
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return ""


def main():
    mirror_dirs = sorted(glob.glob(MIRROR_PATTERN))
    if not mirror_dirs:
        print("No mirror*_data directories found.")
        return

    # Phase 1: collect all records
    all_records = []
    total_scanned = 0

    for mirror_dir in mirror_dirs:
        mirror_name = os.path.basename(mirror_dir)
        mirror_id_match = MIRROR_ID_RE.search(mirror_name)
        mirror_id = mirror_id_match.group(1) if mirror_id_match else mirror_name

        if not os.path.isdir(mirror_dir):
            continue

        for patient_folder in sorted(os.listdir(mirror_dir)):
            if not PATIENT_FOLDER_RE.match(patient_folder):
                continue
            info_path = os.path.join(mirror_dir, patient_folder, "patient_info.txt")
            if not os.path.isfile(info_path):
                continue

            content = None
            for enc in ("utf-8", "gbk"):
                try:
                    with open(info_path, "r", encoding=enc) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if content is None:
                continue

            total_scanned += 1

            # Extract fields
            m = PATIENT_ID_RE.search(content)
            patient_id = m.group(1) if m else patient_folder
            hospital_patient_id = extract_hospital_patient_id(content)
            session_date = extract_session_date(content)

            if not hospital_patient_id or not session_date:
                continue

            # Convert hospital_patient_id to int for grouping
            try:
                hosp_id_int = int(hospital_patient_id)
            except ValueError:
                continue

            all_records.append({
                "patient_id": patient_id,
                "hospital_patient_id": hospital_patient_id,
                "hospital_patient_id_int": hosp_id_int,
                "mirror_id": mirror_id,
                "session_date": session_date,
                "patient_folder": patient_folder,
                "file_path": info_path,
            })

    # Phase 2: group by (hospital_patient_id_int, session_date)
    groups = defaultdict(list)
    for rec in all_records:
        key = (rec["hospital_patient_id_int"], rec["session_date"])
        groups[key].append(rec)

    # Phase 3: keep only groups with 2+ sessions on the same day
    results = []
    stimulated_keys = set()
    for key, recs in sorted(groups.items()):
        if len(recs) >= 2:
            stimulated_keys.add(key)
            for rec in recs:
                results.append({
                    "patient_id": rec["patient_id"],
                    "hospital_patient_id": rec["hospital_patient_id"],
                    "mirror_id": rec["mirror_id"],
                    "session_date": rec["session_date"],
                    "patient_folder": rec["patient_folder"],
                    "file_path": rec["file_path"],
                })

    # Write CSV
    fieldnames = [
        "patient_id",
        "hospital_patient_id",
        "mirror_id",
        "session_date",
        "patient_folder",
        "file_path",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Scanned {total_scanned} patient_info.txt files across {len(mirror_dirs)} mirror directories.")
    print(f"Found {len(stimulated_keys)} (hospital_patient_id, date) groups with 2+ sessions.")
    print(f"Total {len(results)} records marked as electrical-stimulation patients.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
