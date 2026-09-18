#!/usr/bin/env python3
"""Merge every converted Anzhen lab CSV into one deduplicated table."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


EXPECTED_HEADER = (
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
    "检验套名称",
    "检验项名称",
    "检验值(文本)",
    "单位",
    "标本名称",
    "报告时间",
)


def source_files(input_dir: Path) -> list[Path]:
    report_path = input_dir / "转换核对报告.csv"
    if not report_path.is_file():
        raise RuntimeError(f"缺少转换核对报告: {report_path}")
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["输出文件"] for row in rows if row["源文件"] != "合计"]
    if not names or any(not name for name in names):
        raise RuntimeError("转换核对报告中没有有效输出文件列表")
    paths = [input_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"核对报告列出的 CSV 不存在: {missing}")
    return paths


def merge(input_dir: Path, output_path: Path) -> tuple[int, int, int]:
    paths = source_files(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    writer = csv.writer(temporary)
    seen: set[tuple[str, ...]] = set()
    input_rows = 0
    duplicate_rows = 0
    completed = False
    try:
        writer.writerow(EXPECTED_HEADER)
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                try:
                    header = tuple(next(reader))
                except StopIteration as exc:
                    raise RuntimeError(f"空 CSV: {path}") from exc
                if header != EXPECTED_HEADER:
                    raise RuntimeError(f"表头不一致: {path}")
                for line_number, row in enumerate(reader, start=2):
                    input_rows += 1
                    if len(row) != len(EXPECTED_HEADER):
                        raise RuntimeError(
                            f"{path} 第 {line_number} 行不是 18 列"
                        )
                    key = tuple(row)
                    if key in seen:
                        duplicate_rows += 1
                        continue
                    seen.add(key)
                    writer.writerow(row)
        temporary.flush()
        os.fsync(temporary.fileno())
        completed = True
    finally:
        temporary.close()
        if not completed:
            temporary_path.unlink(missing_ok=True)

    os.replace(temporary_path, output_path)
    output_rows = len(seen)
    if input_rows != output_rows + duplicate_rows:
        raise RuntimeError("合并行数不守恒")
    return input_rows, output_rows, duplicate_rows


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "260917安贞化验" / "转换结果",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "merged_lab_tests.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_rows, output_rows, duplicate_rows = merge(
        args.input_dir.resolve(), args.output.resolve()
    )
    print(
        f"合并完成: 输入 {input_rows} 条，去重 {duplicate_rows} 条，"
        f"输出 {output_rows} 条 -> {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
