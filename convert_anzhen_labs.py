#!/usr/bin/env python3
"""Convert Anzhen wide-form laboratory XLSX exports to validated long CSV files.

The source workbooks contain three header rows and one row per encounter.  Every
laboratory block has seven columns, and multiple measurements in a cell are
joined with ``^``.  A literal ``^`` may also be part of a unit (for example
``*10^9/L``), so blindly splitting every cell produces shifted/corrupted rows.

The generated CSV files use the useful (second) header row of the existing
``乳酸.csv`` as their schema.  Its invalid first descriptive row and trailing
empty column are intentionally not emitted.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

SOURCE_LAB_FIELDS = (
    "检验套名称",
    "检验项名称",
    "采集时间",
    "检验值(文本)",
    "单位",
    "标本名称",
    "报告时间",
)
METADATA_FIELDS = (
    "首页病案号",
    "首页性别",
    "首页就诊时年龄",
    "首页入院时间",
    "首页入院科室",
    "首页出院时间",
    "首页出院科室",
    "首页住院天数",
    "手术开始日期",
    "手术结束日期",
    "首页手术操作名称",
    "首页离院方式",
)
OUTPUT_LAB_FIELDS = (
    "检验套名称",
    "检验项名称",
    "检验值(文本)",
    "单位",
    "标本名称",
    "报告时间",
)
OUTPUT_FIELDS = METADATA_FIELDS + OUTPUT_LAB_FIELDS
UNMATCHED_FIELDS = (
    "源文件",
    "源工作表行",
    "患者索引",
    "登记号就诊号",
    "生成的首页病案号",
) + SOURCE_LAB_FIELDS + ("处理说明",)
SUMMARY_FIELDS = (
    "源文件",
    "输出文件",
    "源就诊行数",
    "化验列组数",
    "化验记录总数",
    "原行已有病案首页数",
    "按就诊号补全并复用病案信息数",
    "按就诊号补全但仅有病案号数",
    "最终输出数",
    "不完整条目或组数(无报告时间)",
    "输出内完全重复数",
    "检验项种数",
)
ITEM_COUNT_FIELDS = ("源文件", "检验项名称", "输出记录数")

CELL_REF_RE = re.compile(r"^([A-Z]+)")


class ConversionError(RuntimeError):
    """Raised when an input would otherwise result in a silent misalignment."""


@dataclass(frozen=True)
class WorkbookLayout:
    metadata_start: int
    lab_group_count: int
    first_lab_description: str


@dataclass
class ConversionStats:
    source_file: str
    output_file: str
    source_encounter_rows: int = 0
    lab_group_count: int = 0
    total_measurements: int = 0
    matched_measurements: int = 0
    direct_metadata_measurements: int = 0
    reused_metadata_measurements: int = 0
    id_only_measurements: int = 0
    incomplete_entries: int = 0
    exact_duplicate_rows: int = 0
    item_counts: Counter[str] = field(default_factory=Counter)

    def summary_row(self) -> list[str | int]:
        return [
            self.source_file,
            self.output_file,
            self.source_encounter_rows,
            self.lab_group_count,
            self.total_measurements,
            self.direct_metadata_measurements,
            self.reused_metadata_measurements,
            self.id_only_measurements,
            self.matched_measurements,
            self.incomplete_entries,
            self.exact_duplicate_rows,
            len(self.item_counts),
        ]


def column_index(cell_reference: str) -> int:
    """Return the zero-based column index from an Excel A1 reference."""
    match = CELL_REF_RE.match(cell_reference)
    if not match:
        raise ConversionError(f"无效的 Excel 单元格地址: {cell_reference!r}")
    result = 0
    for char in match.group(1):
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def load_shared_strings(workbook: ZipFile) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in workbook.namelist():
        return []
    with workbook.open(name) as stream:
        root = ElementTree.parse(stream).getroot()
    return [
        "".join(node.text or "" for node in item.iter(XML_NS + "t"))
        for item in root
    ]


def cell_text(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(XML_NS + "t"))

    value = cell.find(XML_NS + "v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)]
        except (ValueError, IndexError) as exc:
            raise ConversionError("共享字符串索引无效") from exc
    if cell_type == "b":
        return "TRUE" if value.text == "1" else "FALSE"
    return value.text


def worksheet_rows(path: Path) -> Iterator[tuple[int, dict[int, str]]]:
    """Stream the first worksheet without requiring pandas/openpyxl."""
    try:
        workbook = ZipFile(path)
    except (BadZipFile, OSError) as exc:
        raise ConversionError(f"无法打开工作簿 {path}") from exc

    with workbook:
        shared_strings = load_shared_strings(workbook)
        worksheet_names = sorted(
            name
            for name in workbook.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheet_names:
            raise ConversionError(f"{path.name} 中没有工作表")
        # The directory also contains a multi-sheet auxiliary patient-ID list.
        # Its first sheet is sufficient to classify it as non-laboratory data.
        # All actual laboratory exports in this batch store their wide table in
        # the first worksheet and are subsequently validated by inspect_layout.
        with workbook.open(worksheet_names[0]) as stream:
            for _event, element in ElementTree.iterparse(stream, events=("end",)):
                if element.tag != XML_NS + "row":
                    continue
                try:
                    row_number = int(element.get("r", "0"))
                except ValueError as exc:
                    raise ConversionError(f"{path.name} 中存在无效行号") from exc
                values = {
                    column_index(cell.get("r", "")): cell_text(cell, shared_strings)
                    for cell in element.findall(XML_NS + "c")
                }
                yield row_number, values
                element.clear()


def inspect_layout(
    path: Path,
    first_header: dict[int, str],
    second_header: dict[int, str],
    field_header: dict[int, str],
) -> WorkbookLayout | None:
    """Recognize and strictly validate the Anzhen laboratory export layout."""
    metadata_columns = [
        index for index, value in first_header.items() if value == "病案首页"
    ]
    if not metadata_columns:
        return None
    metadata_start = min(metadata_columns)
    lab_column_count = metadata_start - 1  # Column A is the patient index.
    if lab_column_count <= 0 or lab_column_count % len(SOURCE_LAB_FIELDS):
        raise ConversionError(
            f"{path.name}: 化验区共有 {lab_column_count} 列，不能拆成 7 列一组"
        )
    lab_group_count = lab_column_count // len(SOURCE_LAB_FIELDS)

    for group_index in range(lab_group_count):
        start = 1 + group_index * len(SOURCE_LAB_FIELDS)
        actual = tuple(field_header.get(start + offset, "") for offset in range(7))
        if actual != SOURCE_LAB_FIELDS:
            raise ConversionError(
                f"{path.name}: 第 {group_index + 1} 个化验组字段异常: {actual!r}"
            )

        descriptions = {
            second_header.get(start + offset, "") for offset in range(7)
        }
        if len(descriptions) != 1 or not next(iter(descriptions)):
            raise ConversionError(
                f"{path.name}: 第 {group_index + 1} 个化验组的标题不一致"
            )

    actual_metadata = tuple(
        field_header.get(metadata_start + offset, "")
        for offset in range(len(METADATA_FIELDS))
    )
    if actual_metadata != METADATA_FIELDS:
        raise ConversionError(
            f"{path.name}: 病案首页字段异常: {actual_metadata!r}"
        )
    if field_header.get(metadata_start + len(METADATA_FIELDS), "") != "regno_admno":
        raise ConversionError(f"{path.name}: 缺少用于核对就诊匹配的 regno_admno")

    return WorkbookLayout(
        metadata_start=metadata_start,
        lab_group_count=lab_group_count,
        first_lab_description=second_header.get(1, ""),
    )


def normalize_metadata(metadata: tuple[str, ...]) -> tuple[str, ...]:
    """Turn metadata missing-value markers into genuine empty CSV fields."""
    return tuple("" if value.strip() == "-" else value for value in metadata)


def canonical_hospital_id(value: str) -> str:
    """Normalize only the known optional asterisk wrapper around an ID."""
    value = value.strip()
    match = re.fullmatch(r"\*([0-9]+)\*", value)
    return match.group(1) if match else value


def hospital_id_from_encounter_key(encounter_key: str) -> str:
    """Convert the visit-number suffix in regno_admno to a 10-digit ID."""
    visit_number = encounter_key.rsplit("_", 1)[-1].strip()
    match = re.fullmatch(r"[Pp]?([0-9]+)", visit_number)
    if not match:
        raise ConversionError(
            f"无法从登记号就诊号 {encounter_key!r} 提取数字就诊号"
        )
    digits = match.group(1)
    if len(digits) > 10:
        raise ConversionError(
            f"登记号就诊号 {encounter_key!r} 的就诊号超过 10 位"
        )
    return digits.zfill(10)


def parse_report_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_admission_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def select_existing_metadata(
    metadata_registry: dict[str, tuple[tuple[str, ...], ...]],
    hospital_id: str,
    report_time_text: str,
) -> tuple[str, ...] | None:
    """Select the unambiguous existing episode for a generated hospital ID."""
    candidates = metadata_registry.get(canonical_hospital_id(hospital_id), ())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    report_time = parse_report_time(report_time_text)
    if report_time is None:
        return None
    matching: list[tuple[str, ...]] = []
    for metadata in candidates:
        admission = parse_admission_time(metadata[3])
        discharge = parse_admission_time(metadata[5])
        if admission is None or discharge is None:
            continue
        # Reports in this export can be finalized several hours after formal
        # discharge; one day safely covers the observed reporting delay.
        if admission <= report_time <= discharge + timedelta(days=1):
            matching.append(metadata)
    return matching[0] if len(matching) == 1 else None


def build_metadata_registry(
    source_paths: list[Path],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Read the first lab workbook's complete encounter table as an index."""
    for source_path in source_paths:
        rows = worksheet_rows(source_path)
        try:
            _number1, first_header = next(rows)
            _number2, second_header = next(rows)
            _number3, field_header = next(rows)
        except StopIteration:
            continue
        layout = inspect_layout(
            source_path, first_header, second_header, field_header
        )
        if layout is None:
            continue

        registry_sets: dict[str, set[tuple[str, ...]]] = {}
        for _row_number, row in rows:
            metadata = normalize_metadata(
                tuple(
                    row.get(layout.metadata_start + offset, "")
                    for offset in range(len(METADATA_FIELDS))
                )
            )
            if metadata[0]:
                hospital_id = canonical_hospital_id(metadata[0])
                registry_sets.setdefault(hospital_id, set()).add(metadata)
        return {
            hospital_id: tuple(sorted(metadata_values))
            for hospital_id, metadata_values in registry_sets.items()
        }
    raise ConversionError("无法从化验工作簿建立病案首页索引")


def split_unit_field(raw_value: str, measurement_count: int) -> list[str]:
    """Split a joined unit field while preserving carets used as exponents."""
    if measurement_count == 1:
        return [raw_value]

    # Values in one source block describe the same analyte and normally repeat
    # one unit.  Test that representation before treating carets as separators;
    # this correctly parses e.g. '*10^9/L^*10^9/L'.
    content_length = len(raw_value) - (measurement_count - 1)
    if content_length >= 0 and content_length % measurement_count == 0:
        unit_length = content_length // measurement_count
        unit = raw_value[:unit_length]
        if "^".join([unit] * measurement_count) == raw_value:
            return [unit] * measurement_count

    parts = raw_value.split("^")
    if len(parts) == measurement_count:
        return parts
    raise ConversionError(
        "单位字段无法无歧义拆分（单位中的 ^ 可能是指数符号）: "
        f"记录数={measurement_count}, 原值={raw_value!r}"
    )


def split_lab_block(
    path: Path,
    row_number: int,
    group_index: int,
    raw_values: list[str],
) -> list[tuple[str, ...]]:
    """Explode one seven-column block and prove that all fields stay aligned."""
    if not any(raw_values):
        return []
    measurement_count = len(raw_values[6].split("^"))
    split_fields: list[list[str]] = []
    for field_index, raw_value in enumerate(raw_values):
        if field_index == 4:
            parts = split_unit_field(raw_value, measurement_count)
        else:
            parts = raw_value.split("^")
        if len(parts) != measurement_count:
            raise ConversionError(
                f"{path.name} 第 {row_number} 行、第 {group_index + 1} 组的"
                f"{SOURCE_LAB_FIELDS[field_index]}有 {len(parts)} 项，但报告时间有 "
                f"{measurement_count} 项；已停止以防字段错配"
            )
        split_fields.append(parts)

    records = list(zip(*split_fields, strict=True))
    for record_index, record in enumerate(records, start=1):
        # Some historical rows legitimately have no test-suite name (their
        # source column description is e.g. ``_快速C-反应蛋白``).  The item and
        # report time are the fields that identify a laboratory result.
        if not record[1]:
            raise ConversionError(
                f"{path.name} 第 {row_number} 行、第 {group_index + 1} 组、第 "
                f"{record_index} 条缺少检验项"
            )
    return records


def atomic_csv_writer(path: Path):
    """Return a temporary file and writer; caller atomically replaces on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    return handle, csv.writer(handle)


def validate_output_csv(path: Path, expected_rows: int) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ConversionError(f"输出文件 {path} 为空") from exc
        if header != OUTPUT_FIELDS:
            raise ConversionError(f"输出文件 {path} 的表头校验失败")
        actual_rows = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(OUTPUT_FIELDS):
                raise ConversionError(
                    f"输出文件 {path} 第 {line_number} 行不是 18 列"
                )
            actual_rows += 1
    if actual_rows != expected_rows:
        raise ConversionError(
            f"输出文件 {path} 有 {actual_rows} 行，预期 {expected_rows} 行"
        )


def convert_workbook(
    source_path: Path,
    output_path: Path,
    unmatched_writer: csv.writer,
    encounter_metadata: dict[str, tuple[str, ...]],
    metadata_registry: dict[str, tuple[tuple[str, ...], ...]],
) -> ConversionStats | None:
    rows = worksheet_rows(source_path)
    try:
        _row1_number, first_header = next(rows)
        _row2_number, second_header = next(rows)
        _row3_number, field_header = next(rows)
    except StopIteration as exc:
        raise ConversionError(f"{source_path.name}: 工作簿少于三行") from exc

    layout = inspect_layout(source_path, first_header, second_header, field_header)
    if layout is None:
        return None

    stats = ConversionStats(
        source_file=source_path.name,
        output_file=output_path.name,
        lab_group_count=layout.lab_group_count,
    )
    output_handle, output_writer = atomic_csv_writer(output_path)
    temporary_path = Path(output_handle.name)
    duplicate_counter: Counter[tuple[str, ...]] = Counter()
    completed = False
    try:
        output_writer.writerow(OUTPUT_FIELDS)
        for row_number, row in rows:
            stats.source_encounter_rows += 1
            patient_index = row.get(0, "").strip()
            metadata = normalize_metadata(
                tuple(
                    row.get(layout.metadata_start + offset, "")
                    for offset in range(len(METADATA_FIELDS))
                )
            )
            encounter_key = row.get(
                layout.metadata_start + len(METADATA_FIELDS), ""
            ).strip()

            if encounter_key and patient_index and not encounter_key.startswith(
                patient_index + "_"
            ):
                raise ConversionError(
                    f"{source_path.name} 第 {row_number} 行患者索引 {patient_index!r} "
                    f"与就诊键 {encounter_key!r} 不一致"
                )
            if metadata[0] and not encounter_key:
                raise ConversionError(
                    f"{source_path.name} 第 {row_number} 行有病案首页但无就诊键"
                )
            if metadata[0]:
                previous = encounter_metadata.setdefault(encounter_key, metadata)
                if previous != metadata:
                    raise ConversionError(
                        f"就诊键 {encounter_key!r} 在不同源表中对应了不同病案首页"
                    )

            for group_index in range(layout.lab_group_count):
                group_start = 1 + group_index * len(SOURCE_LAB_FIELDS)
                raw_values = [row.get(group_start + offset, "") for offset in range(7)]
                if any(raw_values) and not raw_values[6]:
                    # Historical exports can contain an order/placeholder but
                    # no report.  Preserve it in the audit file, but do not
                    # pretend it is a reported laboratory result.
                    stats.incomplete_entries += 1
                    unmatched_writer.writerow(
                        [
                            source_path.name,
                            row_number,
                            patient_index,
                            encounter_key,
                            "",
                            *raw_values,
                            "缺少报告时间，未计入正式化验记录",
                        ]
                    )
                    continue
                records = split_lab_block(
                    source_path, row_number, group_index, raw_values
                )

                for record in records:
                    if not record[6]:
                        stats.incomplete_entries += 1
                        unmatched_writer.writerow(
                            [
                                source_path.name,
                                row_number,
                                patient_index,
                                encounter_key,
                                "",
                                *record,
                                "拆分后该条目缺少报告时间，未计入正式化验记录",
                            ]
                        )
                        continue
                    stats.total_measurements += 1
                    if metadata[0]:
                        resolved_metadata = metadata
                        stats.direct_metadata_measurements += 1
                    else:
                        generated_id = hospital_id_from_encounter_key(encounter_key)
                        existing_metadata = select_existing_metadata(
                            metadata_registry, generated_id, record[6]
                        )
                        if existing_metadata is not None:
                            # Keep the requested generated, zero-padded ID while
                            # reusing the remaining fields from the matched page.
                            resolved_metadata = (generated_id,) + existing_metadata[1:]
                            stats.reused_metadata_measurements += 1
                            handling = "由就诊号生成病案号，并复用已有病案首页信息"
                        else:
                            resolved_metadata = (generated_id,) + ("",) * (
                                len(METADATA_FIELDS) - 1
                            )
                            stats.id_only_measurements += 1
                            handling = "由就诊号生成病案号；未找到可复用病案首页，其余字段留空"
                        unmatched_writer.writerow(
                            [
                                source_path.name,
                                row_number,
                                patient_index,
                                encounter_key,
                                generated_id,
                                *record,
                                handling,
                            ]
                        )

                    # Omit collection time to match the requested 18-column schema.
                    output_record = resolved_metadata + (
                        record[0],
                        record[1],
                        record[3],
                        record[4],
                        record[5],
                        record[6],
                    )
                    output_writer.writerow(output_record)
                    duplicate_counter[output_record] += 1
                    stats.item_counts[record[1]] += 1
                    stats.matched_measurements += 1

        if stats.total_measurements != stats.matched_measurements:
            raise ConversionError(f"{source_path.name}: 内部记录计数不守恒")
        if stats.matched_measurements != (
            stats.direct_metadata_measurements
            + stats.reused_metadata_measurements
            + stats.id_only_measurements
        ):
            raise ConversionError(f"{source_path.name}: 病案信息来源计数不守恒")
        stats.exact_duplicate_rows = sum(
            count - 1 for count in duplicate_counter.values() if count > 1
        )
        output_handle.flush()
        os.fsync(output_handle.fileno())
        completed = True
    finally:
        output_handle.close()
        if not completed:
            temporary_path.unlink(missing_ok=True)

    os.replace(temporary_path, output_path)
    validate_output_csv(output_path, stats.matched_measurements)
    return stats


def output_name(source_path: Path) -> str:
    stem = source_path.stem
    if stem.startswith("健康镜"):
        stem = stem[len("健康镜") :]
    return stem + ".csv"


def write_audit_files(
    output_dir: Path,
    stats_list: list[ConversionStats],
    skipped_files: list[str],
    unmatched_temp_path: Path,
) -> None:
    summary_path = output_dir / "转换核对报告.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_FIELDS)
        for stats in stats_list:
            writer.writerow(stats.summary_row())
        total = ConversionStats(
            source_file="合计",
            output_file="",
            source_encounter_rows=sum(x.source_encounter_rows for x in stats_list),
            lab_group_count=sum(x.lab_group_count for x in stats_list),
            total_measurements=sum(x.total_measurements for x in stats_list),
            matched_measurements=sum(x.matched_measurements for x in stats_list),
            direct_metadata_measurements=sum(
                x.direct_metadata_measurements for x in stats_list
            ),
            reused_metadata_measurements=sum(
                x.reused_metadata_measurements for x in stats_list
            ),
            id_only_measurements=sum(x.id_only_measurements for x in stats_list),
            incomplete_entries=sum(x.incomplete_entries for x in stats_list),
            exact_duplicate_rows=sum(x.exact_duplicate_rows for x in stats_list),
        )
        total.item_counts.update(
            item for stats in stats_list for item in stats.item_counts
        )
        writer.writerow(total.summary_row())

    item_count_path = output_dir / "检验项计数.csv"
    with item_count_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ITEM_COUNT_FIELDS)
        for stats in stats_list:
            for item_name, count in sorted(stats.item_counts.items()):
                writer.writerow([stats.source_file, item_name, count])

    unmatched_path = output_dir / "未匹配化验记录.csv"
    os.replace(unmatched_temp_path, unmatched_path)

    skipped_path = output_dir / "跳过的辅助文件.txt"
    skipped_path.write_text("\n".join(skipped_files) + "\n", encoding="utf-8")


def convert_directory(input_dir: Path, output_dir: Path) -> list[ConversionStats]:
    if not input_dir.is_dir():
        raise ConversionError(f"输入目录不存在: {input_dir}")
    source_paths = sorted(
        path
        for path in input_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
    )
    if not source_paths:
        raise ConversionError(f"输入目录中没有 .xlsx 文件: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_registry = build_metadata_registry(source_paths)
    print(
        f"已建立病案首页索引: {len(metadata_registry)} 个病案号",
        flush=True,
    )
    unmatched_handle, unmatched_writer = atomic_csv_writer(
        output_dir / "未匹配化验记录.csv"
    )
    unmatched_temp_path = Path(unmatched_handle.name)
    unmatched_writer.writerow(UNMATCHED_FIELDS)

    stats_list: list[ConversionStats] = []
    skipped_files: list[str] = []
    encounter_metadata: dict[str, tuple[str, ...]] = {}
    completed = False
    try:
        for source_path in source_paths:
            target_path = output_dir / output_name(source_path)
            stats = convert_workbook(
                source_path,
                target_path,
                unmatched_writer,
                encounter_metadata,
                metadata_registry,
            )
            if stats is None:
                skipped_files.append(source_path.name)
                print(f"跳过辅助文件: {source_path.name}", flush=True)
                continue
            stats_list.append(stats)
            print(
                f"已转换 {source_path.name} -> {target_path.name}: "
                f"输出 {stats.matched_measurements} 条，"
                f"其中就诊号补全 {stats.reused_metadata_measurements + stats.id_only_measurements} 条",
                flush=True,
            )

        if not stats_list:
            raise ConversionError("没有识别到符合结构的化验工作簿")
        unmatched_handle.flush()
        os.fsync(unmatched_handle.fileno())
        completed = True
    finally:
        unmatched_handle.close()
        if not completed:
            unmatched_temp_path.unlink(missing_ok=True)

    write_audit_files(
        output_dir,
        stats_list,
        skipped_files,
        unmatched_temp_path,
    )
    return stats_list


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="将安贞化验宽表 XLSX 严格转换为 18 列长表 CSV。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "260917安贞化验",
        help="源 XLSX 目录（默认: %(default)s）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "260917安贞化验" / "转换结果",
        help="输出目录（默认: %(default)s）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats_list = convert_directory(args.input_dir.resolve(), args.output_dir.resolve())
    except ConversionError as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 1

    total = sum(stats.matched_measurements for stats in stats_list)
    supplemented = sum(
        stats.reused_metadata_measurements + stats.id_only_measurements
        for stats in stats_list
    )
    reused = sum(stats.reused_metadata_measurements for stats in stats_list)
    print(
        f"转换完成: {len(stats_list)} 个化验文件，共输出 {total} 条；"
        f"按就诊号补全 {supplemented} 条（复用已有病案信息 {reused} 条）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
