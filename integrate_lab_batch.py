#!/usr/bin/env python3
"""Audit a converted batch and optionally append non-repeated rows safely."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from convert_anzhen_labs import canonical_hospital_id
from merge_converted_labs import EXPECTED_HEADER, source_files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(values: tuple[str, ...]) -> bytes:
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    ).digest()


def hospital_id(value: str) -> str:
    value = canonical_hospital_id(value)
    return value.zfill(10) if value.isdigit() and len(value) <= 10 else value


def read_rows(path: Path, require_identity: bool = False):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        if tuple(next(reader)) != EXPECTED_HEADER:
            raise ValueError(f"Unexpected CSV header: {path}")
        for line, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_HEADER) or (
                require_identity and (not row[0] or not row[13] or not row[17])
            ):
                raise ValueError(f"Invalid lab record: {path}:{line}")
            yield line, tuple(row)


def lab_key(row: tuple[str, ...]) -> bytes:
    # The page's demographic/surgical metadata can change without creating a
    # new laboratory result. Retain all six lab fields, including value/unit.
    return digest((hospital_id(row[0]),) + row[12:])


def identity_key(row: tuple[str, ...]) -> bytes:
    return digest((hospital_id(row[0]), row[12], row[13], row[16], row[17]))


def write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def integrate(input_dir: Path, table: Path, apply: bool = False) -> dict:
    before_hash = sha256(table)
    before_bytes = table.stat().st_size
    old_full = set()
    old_labs = set()
    old_identity = defaultdict(set)
    old_ids = set()
    old_rows = 0
    for _, row in read_rows(table):
        old_rows += 1
        old_full.add(digest(row))
        old_labs.add(lab_key(row))
        old_identity[identity_key(row)].add((row[14], row[15]))
        old_ids.add(hospital_id(row[0]))
    print(f"Indexed existing table: {old_rows} rows", flush=True)

    seen = set()
    batch_ids = set()
    novel_ids = set()
    additions = []
    summaries = []
    differences = []
    sources = []
    for path in source_files(input_dir):
        count = Counter()
        sources.append({"file": path.name, "sha256": sha256(path)})
        for line, row in read_rows(path, require_identity=True):
            count["input"] += 1
            batch_ids.add(hospital_id(row[0]))
            key = lab_key(row)
            if key in seen:
                count["batch_duplicates"] += 1
                continue
            seen.add(key)
            old_values = old_identity.get(identity_key(row), set())
            if digest(row) in old_full:
                count["exact_existing_duplicates"] += 1
                continue
            if key in old_labs:
                count["metadata_only_duplicates"] += 1
                category = "化验内容相同，仅病案信息或病案号包装不同，保留已有记录"
            elif any(
                (not row[14] or row[14] == value)
                and (not row[15] or row[15] == unit)
                for value, unit in old_values
            ):
                count["less_complete_duplicates"] += 1
                category = "新批结果或单位缺失，已有同一检测信息更完整，保留已有记录"
            else:
                count["additions"] += 1
                additions.append(row)
                novel_ids.add(hospital_id(row[0]))
                if not old_values:
                    continue
                if any(value == row[14] and unit != row[15] for value, unit in old_values):
                    category = "同一检测值、不同原始单位，保留新单位表示"
                elif any(not value or not unit for value, unit in old_values):
                    category = "已有检测缺少结果或单位，追加本批补充信息"
                    count["supplemented_results"] += 1
                elif all(unit != row[15] for value, unit in old_values):
                    category = "同一检测时点、不同原始单位，保留两种表示"
                    count["alternative_units"] += 1
                else:
                    category = "同一检测时点、同一单位、不同结果，需要人工核对"
                    count["unresolved_conflicts"] += 1
            differences.append((
                category, path.name, line,
                json.dumps(sorted(old_values), ensure_ascii=False), *row,
            ))
        summaries.append({"file": path.name, **dict(count)})
        print(f"{path.name}: {dict(count)}", flush=True)

    total = Counter()
    for entry in summaries:
        total.update({key: value for key, value in entry.items() if key != "file"})
    accounted = sum(total[key] for key in (
        "batch_duplicates", "exact_existing_duplicates", "metadata_only_duplicates",
        "less_complete_duplicates", "additions",
    ))
    if accounted != total["input"]:
        raise ValueError("Input/duplicate/addition accounting failed")
    if sha256(table) != before_hash:
        raise ValueError("Existing table changed during the audit")

    now = datetime.now(timezone.utc)
    report = {
        "created_at_utc": now.isoformat(),
        "operation": "preserve existing CSV bytes; append batch lab-content union",
        "duplicate_key": "normalized hospital ID + all six original lab fields",
        "measurement_identity": "normalized hospital ID + suite + item + specimen + report time",
        "alternative_units": "preserved without conversion; not additional independent test events",
        "old_table": str(table), "old_rows": old_rows,
        "old_sha256": before_hash, "old_patients": len(old_ids),
        "old_exact_duplicate_rows_preserved": old_rows - len(old_full),
        "counts": dict(total), "per_file": summaries, "sources": sources,
        "batch_patients": len(batch_ids),
        "patients_already_in_table": len(batch_ids & old_ids),
        "new_patients": len(novel_ids - old_ids),
        "expected_final_rows": old_rows + len(additions),
        "applied": False,
    }
    write_csv(input_dir / "新增化验记录.csv", EXPECTED_HEADER, additions)
    write_csv(
        input_dir / "差异核对记录.csv",
        ("处理说明", "源CSV", "源CSV行", "已有结果与单位", *EXPECTED_HEADER), differences,
    )
    summary_keys = (
        "input", "batch_duplicates", "exact_existing_duplicates", "metadata_only_duplicates",
        "less_complete_duplicates", "additions", "alternative_units",
        "supplemented_results", "unresolved_conflicts",
    )
    write_csv(
        input_dir / "重复核对报告.csv", ("file", *summary_keys),
        [tuple([entry["file"]] + [entry.get(key, 0) for key in summary_keys]) for entry in summaries]
        + [tuple(["TOTAL"] + [total[key] for key in summary_keys])],
    )
    report_path = input_dir / "合并核对报告.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if total["unresolved_conflicts"]:
        raise ValueError(f"Unresolved result conflicts: {total['unresolved_conflicts']}; see difference audit")

    if apply and additions:
        backup_dir = table.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_dir / f"{table.stem}.before_{input_dir.parent.name}_{stamp}{table.suffix}"
        # Exclusive creation prevents overwriting any previous backup.
        with table.open("rb") as source, backup.open("xb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(table, backup)
        if sha256(backup) != before_hash:
            raise ValueError("Backup checksum mismatch; original table untouched")
        fd, temp_name = tempfile.mkstemp(prefix=f".{table.name}.", suffix=".tmp", dir=table.parent)
        os.close(fd)
        temporary = Path(temp_name)
        try:
            shutil.copyfile(backup, temporary)
            with temporary.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) not in (b"\n", b"\r"):
                    raise ValueError("Existing CSV has no terminal newline")
            with temporary.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(additions)
                f.flush()
                os.fsync(f.fileno())
            final_count = sum(1 for _ in read_rows(temporary))
            if final_count != report["expected_final_rows"]:
                raise ValueError("Final CSV row count mismatch")
            prefix_hash = hashlib.sha256()
            with temporary.open("rb") as f:
                remaining = before_bytes
                while remaining:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated original CSV prefix")
                    prefix_hash.update(chunk)
                    remaining -= len(chunk)
            if prefix_hash.hexdigest() != before_hash or sha256(table) != before_hash:
                raise ValueError("Original records changed; original table untouched")
            shutil.copymode(table, temporary)
            final_hash = sha256(temporary)
            os.replace(temporary, table)
            report.update(applied=True, backup=str(backup), final_rows=final_count, final_sha256=final_hash)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps({key: report[key] for key in (
        "old_rows", "counts", "batch_patients", "new_patients", "expected_final_rows", "applied",
    )}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--table", type=Path, default=Path("merged_lab_tests.csv"))
    parser.add_argument("--apply", action="store_true", help="Back up and append after all checks pass")
    args = parser.parse_args()
    integrate(args.input_dir.resolve(), args.table.resolve(), args.apply)


if __name__ == "__main__":
    main()
