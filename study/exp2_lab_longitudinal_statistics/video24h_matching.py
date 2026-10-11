"""Rank all current laboratory fields by same-admission 24h video coverage."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.face_video import resolve_video, scan_ffv1
from study.common.time_alignment import local_naive_to_unix, read_session_metadata, TimeAlignmentError
from study.exp2_face_history_head32_regression.source_data import _nearest_measurement
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.config import FRAME_QUANTILES, MIN_SOURCE_FRAME_GAP, DATA_ROOT
from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id
from .run_analysis import _harmonize_analytes, _normalize_unit, _sha256, _join_unique, _figure_style


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB = ROOT / "merged_lab_tests.csv"
REFERENCE = HERE.parent / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
TABLES = HERE / "outputs/tables/video24h_matching"
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RAW_KEYS = ["检验项名称", "单位", "标本名称"]


def parse_times(series):
    """Mixed local date formats; explicit offsets remain absolute timestamps."""
    unique = pd.Series(series.astype(str).str.strip().unique())
    explicit_zone = unique.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$", regex=True)
    values = pd.Series(np.nan, index=unique.index)
    for aware, selected in ((False, ~explicit_zone), (True, explicit_zone)):
        parsed = pd.to_datetime(unique[selected], format="mixed", errors="coerce", utc=True if aware else False)
        if not aware:
            parsed = parsed.dt.tz_localize("Asia/Shanghai", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        values.loc[selected] = (parsed.astype("int64") / 1e9).where(parsed.notna())
    return series.astype(str).str.strip().map(dict(zip(unique, values)))


def exact_numeric(series):
    text = series.astype(str).str.strip().str.replace("−", "-", regex=False)
    comma = text.str.contains(",", regex=False)
    thousands = text.str.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    text = text.mask(comma & ~thousands, "").str.replace(",", "", regex=False)
    valid = text.str.fullmatch(NUMBER)
    value = pd.to_numeric(text.where(valid), errors="coerce")
    return value.where(np.isfinite(value))


def specimen_group(specimen, item):
    specimen = str(specimen).strip()
    gas = any(token in item for token in ("氧分压", "二氧化碳", "氧饱和", "氧合血红", "碳氧血红", "还原血红",
                                          "高铁血红", "肺泡", "碱剩余", "碳酸氢", "酸碱度", "PH7.4"))
    if "动脉血" in specimen:
        return "动脉血" if gas else "血液"
    if "静脉血" in specimen:
        return "静脉血" if gas else "血液"
    if specimen in {"血", "血清", "血浆", "全血"}:
        return "血液（动静脉未注明）" if gas else "血液"
    if "尿" in specimen:
        return "24小时尿" if "24" in specimen else "尿"
    return specimen or "标本未注明"


def family_name(item):
    families = (
        ("组织损伤酶", ("乳酸脱氢酶", "肌酸激酶", "羟丁酸脱氢酶")),
        ("酸碱与通气", ("酸碱度",)),
        ("糖代谢", ("糖化", "葡萄糖", "血糖")),
        ("胆红素", ("胆红素",)),
        ("肌钙蛋白", ("肌钙蛋白",)),
        ("心脏利钠肽", ("BNP", "钠尿肽")),
        ("氧合与氧亲和力", ("氧分压", "肺泡", "氧饱和")),
        ("血红蛋白及其组分", ("血红蛋白",)),
        ("红细胞计数及形态", ("红细胞",)),
        ("肾功能", ("肌酐", "eGFR", "胱抑素", "尿素", "尿酸")),
        ("凝血与血小板", ("血小板", "凝血", "纤维蛋白", "D-二聚体", "INR")),
        ("炎症与白细胞", ("反应蛋白", "降钙素", "白细胞", "粒细胞", "淋巴细胞", "单核细胞", "嗜酸", "嗜碱")),
        ("酸碱与通气", ("二氧化碳", "碱剩余", "碳酸氢", "酸碱度", "阴离子间隙")),
        ("乳酸", ("乳酸",)),
        ("蛋白与肝酶", ("白蛋白", "球蛋白", "总蛋白", "转氨酶", "磷酸酶", "肽酶", "胆碱酯酶")),
        ("脂代谢", ("胆固醇", "甘油三酯", "脂蛋白")),
        ("电解质", ("钠", "钾", "氯", "钙", "镁", "磷")),
    )
    for name, tokens in families:
        if any(token in item for token in tokens):
            return name
    return "其他：" + item


def load_labs():
    columns = ["首页病案号", "首页入院时间", "首页出院时间", "检验套名称", "报告时间", "检验值(文本)"] + RAW_KEYS
    raw = pd.read_csv(LAB, dtype=str, keep_default_na=False, usecols=columns)
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["report_unix"] = parse_times(raw["报告时间"])
    raw["admission_unix"] = parse_times(raw["首页入院时间"])
    raw["discharge_unix"] = parse_times(raw["首页出院时间"])
    raw["numeric_value"] = exact_numeric(raw["检验值(文本)"])
    raw["base_valid"] = raw.hospital_id.ne("") & raw.report_unix.notna() & raw["检验项名称"].str.strip().ne("")
    raw["any_valid"] = raw.base_valid & raw["检验值(文本)"].str.strip().ne("")
    raw["numeric_valid"] = raw.base_valid & raw.numeric_value.notna()
    raw["item_name"] = raw["检验项名称"].str.strip()
    raw["unit"] = raw["单位"].map(_normalize_unit)
    harmonized, audit = _harmonize_analytes(raw)
    harmonized["specimen_group"] = [specimen_group(specimen, item) for specimen, item in zip(harmonized["标本名称"], harmonized.item_name)]
    harmonized["family"] = harmonized.item_name.map(family_name)
    for column in ("source_id", "canonical_id"):
        keys = RAW_KEYS if column == "source_id" else ["item_name", "unit", "specimen_group"]
        unique = sorted(set(map(tuple, harmonized[keys].to_numpy())))
        lookup = {key: f"{'F' if column == 'source_id' else 'C'}{i+1:04d}" for i, key in enumerate(unique)}
        harmonized[column] = [lookup[key] for key in map(tuple, harmonized[keys].to_numpy())]
    episodes = raw.loc[raw.hospital_id.ne("") & raw.admission_unix.notna() & raw.discharge_unix.gt(raw.admission_unix),
                       ["hospital_id", "admission_unix", "discharge_unix"]].drop_duplicates()
    return harmonized, episodes, audit


def video_inventory(episodes):
    quality = json.loads((REFERENCE / "source_data/data_quality_report.json").read_text())
    signature = quality["source_fingerprints"]["merged_lab_tests"]
    if signature["size_bytes"] != LAB.stat().st_size or signature["mtime_ns"] != LAB.stat().st_mtime_ns:
        raise RuntimeError("Current Exp2 video-session audit was built for a different lab table")
    audit = pd.read_csv(REFERENCE / "source_data/raw_video_audit.csv", dtype={"hospital_id": str, "video_id": str})
    patients = {patient: group[["admission_unix", "discharge_unix"]].to_numpy() for patient, group in episodes.groupby("hospital_id")}
    rows = []
    for row in audit.itertuples(index=False):
        if row.status not in {"retained_24h_pool", "supported_lab_outside_24h", "patient_without_supported_lab"}:
            continue
        mirror, local = row.video_id.split("_patient_")
        session = read_session_metadata(Path(row.video_path).parent / "patient_info.txt", expected_local_id=int(local), expected_hospital_id=row.hospital_id)
        if abs(session["session_time_unix"] - row.capture_start_unix) > 1e-6:
            raise RuntimeError(f"Cached video session timestamp changed: {row.video_id}")
        matches = [(a, b) for a, b in patients.get(row.hospital_id, ()) if a <= row.capture_start_unix and row.capture_end_unix <= b]
        if len(matches) != 1:
            raise RuntimeError(f"Cached hospitalization membership changed: {row.video_id}")
        rows.append({"hospital_id": row.hospital_id, "video_id": row.video_id, "mirror": mirror, "lab_patient_id": int(local),
                     "capture_start_unix": row.capture_start_unix, "capture_end_unix": row.capture_end_unix,
                     "admission_unix": matches[0][0], "discharge_unix": matches[0][1]})
    videos = pd.DataFrame(rows)
    if videos.video_id.duplicated().any():
        raise RuntimeError("Duplicate video sessions")
    return videos, audit.status.value_counts().to_dict()


def native_frame_eligibility(videos):
    cached = set()
    directories = ("common/cache/face224_20frame_main", "exp9_face_interpolated_lab_regression/cache/combined_frames20",
                   "common/cache/lab_update_20261007", "exp4/cache/lab_update_20261007/frames20")
    reused = []
    for relative in directories:
        directory = HERE.parent / relative
        path = directory / "frame_offsets.npz"
        if not path.exists():
            continue
        index = FrameOffsetIndex.load(path)
        if set(index.video_formats) == {"ffv1"} and _index_is_reusable(directory, index.video_ids, "20frame"):
            cached.update(str(video) for video in index.video_ids if index.frame_range(video)[1] - index.frame_range(video)[0] == 20)
            reused.append(str(path))
    result = []
    for i, row in enumerate(videos.itertuples(index=False), 1):
        eligible, reason = False, ""
        try:
            path = resolve_video(row, DATA_ROOT)
            if not path.endswith(".mkv"):
                raise RuntimeError("This audit is native224-only; no legacy fallback")
            if row.video_id in cached:
                eligible, reason = True, "verified_existing_20frame_index"
            else:
                starts, _, _, _, failures = scan_ffv1(path, "20frame", FRAME_QUANTILES, MIN_SOURCE_FRAME_GAP)
                eligible = len(starts) == 20
                reason = "verified_20_packet_scan" if eligible else failures[-1]["reason"]
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            reason = str(error)
        result.append({"video_id": row.video_id, "native224_20frame_eligible": eligible, "frame_status": reason})
        if i % 100 == 0:
            print(f"[video-check] {i}/{len(videos)}", flush=True)
    return videos.merge(pd.DataFrame(result), on="video_id", validate="one_to_one"), reused


def nearest_sorted(times, values, video):
    """Only the two neighbors of the midpoint can win the existing tie policy."""
    left = np.searchsorted(times, video.admission_unix, side="left")
    right = np.searchsorted(times, video.discharge_unix, side="right")
    position = np.searchsorted(times, (video.capture_start_unix + video.capture_end_unix) / 2.)
    candidates = [(times[i], values[i]) for i in (position - 1, position) if left <= i < right]
    return _nearest_measurement(candidates, video.capture_start_unix, video.capture_end_unix, 24)


def match_events(labs, videos, key, numeric=True):
    selected = labs.loc[labs.numeric_valid if numeric else labs.any_valid]
    events = selected.groupby([key, "hospital_id", "report_unix"], as_index=False).numeric_value.median()
    by_patient = {patient: list(group.itertuples(index=False)) for patient, group in videos.groupby("hospital_id")}
    rows = []
    for (group_id, patient), group in events.groupby([key, "hospital_id"]):
        if patient not in by_patient:
            continue
        times = group.report_unix.to_numpy(float)
        values = group.numeric_value.to_numpy(float) if numeric else np.ones(len(group))
        for video in by_patient[patient]:
            nearest = nearest_sorted(times, values, video)
            if nearest is None:
                continue
            rows.append({key: group_id, "hospital_id": patient, "video_id": video.video_id, "mirror": video.mirror,
                         "lab_time_unix": nearest["timestamp_unix"], "raw_value": nearest["value"] if numeric else np.nan,
                         "match_delta_h": nearest["delta_h"], "signed_delta_h": nearest["signed_delta_h"],
                         "native224_20frame_eligible": video.native224_20frame_eligible})
    return pd.DataFrame(rows, columns=[key, "hospital_id", "video_id", "mirror", "lab_time_unix", "raw_value", "match_delta_h", "signed_delta_h", "native224_20frame_eligible"])


def ranked_catalog(labs, matches, any_matches, key):
    rows = []
    for group_id, group in labs.groupby(key):
        part = matches.loc[matches[key].eq(group_id)]
        native = part.loc[part.native224_20frame_eligible]
        any_part = any_matches.loc[any_matches[key].eq(group_id)]
        row = {key: group_id, "canonical_name": group.item_name.iloc[0], "canonical_unit": group.unit.iloc[0],
               "specimen_group": group.specimen_group.iloc[0], "family": group.family.iloc[0],
               "source_item_names": _join_unique(group["检验项名称"]), "source_units": _join_unique(group["单位"]),
               "source_specimens": _join_unique(group["标本名称"]), "source_suites": _join_unique(group["检验套名称"]),
               "harmonization_rules": _join_unique(group.harmonization_rule), "source_rows": len(group),
               "valid_numeric_rows": int(group.numeric_valid.sum()), "non_exact_or_invalid_rows": int((~group.numeric_valid).sum()),
               "lab_patients": group.loc[group.numeric_valid].hospital_id.nunique(),
               "lab_events": group.loc[group.numeric_valid, ["hospital_id", "report_unix"]].drop_duplicates().shape[0],
               "matched24h_videos": len(part), "matched24h_patients": part.hospital_id.nunique(),
               "matched24h_distinct_lab_events": part[["hospital_id", "lab_time_unix"]].drop_duplicates().shape[0],
               "native224_20frame_videos": len(native), "native224_20frame_patients": native.hospital_id.nunique(),
               "matched24h_videos_any_result": len(any_part), "source_field_count": group.source_id.nunique()}
        if key == "source_id":
            row.update({column: group[column].iloc[0] for column in RAW_KEYS})
            row["canonical_id"] = group.canonical_id.iloc[0]
            if group.canonical_id.nunique() != 1:
                raise RuntimeError("One exact source field maps to multiple canonical groups")
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["matched24h_videos", "native224_20frame_videos", "canonical_name", key], ascending=[False, False, True, True]).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


def source_table_inventory():
    from merge_seven_csv import CSV_FILES
    from merge_converted_labs import source_files
    rows = [{"path": str(LAB), "role": "authoritative_current_union", "used_in_matching": True,
             "rows": sum(len(chunk) for chunk in pd.read_csv(LAB, usecols=["检验项名称"], chunksize=100000)), "bytes": LAB.stat().st_size}]
    for filename in CSV_FILES:
        path = ROOT / filename
        if path.exists():
            entry = {"path": str(path), "role": "legacy_component_not_added_again", "used_in_matching": False, "bytes": path.stat().st_size}
            errors = []
            for encoding in ("utf-8-sig", "gb18030"):
                for skiprows in (0, 1):
                    try:
                        header = pd.read_csv(path, encoding=encoding, skiprows=skiprows, nrows=0)
                        if "检验项名称" not in header:
                            continue
                        entry.update(rows=sum(len(chunk) for chunk in pd.read_csv(path, encoding=encoding, skiprows=skiprows,
                            usecols=["检验项名称"], chunksize=100000)), encoding=encoding, skipped_description_rows=skiprows)
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError) as error:
                        errors.append(str(error))
                if "rows" in entry:
                    break
            if "rows" not in entry:
                entry["read_error"] = " | ".join(errors) or "No recognized laboratory header"
            rows.append(entry)
    for directory in sorted(ROOT.glob("*安贞化验/转换结果")):
        if not (directory / "转换核对报告.csv").is_file():
            continue
        report = pd.read_csv(directory / "转换核对报告.csv").set_index("输出文件")
        for path in source_files(directory):
            rows.append({"path": str(path), "role": "converted_component_not_added_again", "used_in_matching": False,
                         "rows": int(report.loc[path.name, "最终输出数"]), "bytes": path.stat().st_size})
    for path in sorted((ROOT / "backups").glob("merged_lab_tests*.csv")):
        rows.append({"path": str(path), "role": "historical_backup_excluded", "used_in_matching": False, "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def write_outputs(labs, videos, canonical, source, matches, raw_matches, audit, metadata):
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, table in (("analyte_ranking", canonical), ("source_field_ranking", source), ("video_inventory", videos),
                        ("canonical_video_matches", matches), ("source_video_matches", raw_matches), ("harmonization_audit", audit)):
        table.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    source_table_inventory().to_csv(TABLES / "source_table_inventory.csv", index=False, encoding="utf-8-sig")
    mapping = labs.groupby(["canonical_id", "source_id", "item_name", "unit", "specimen_group", "family", *RAW_KEYS,
                            "harmonization_rule"], as_index=False).size().rename(columns={"size": "source_rows"})
    mapping.to_csv(TABLES / "field_to_analyte_mapping.csv", index=False, encoding="utf-8-sig")
    family_rows = []
    family_lookup = canonical.set_index("canonical_id").family
    enriched = matches.assign(family=matches.canonical_id.map(family_lookup))
    for family, group in canonical.groupby("family"):
        part = enriched.loc[enriched.family.eq(family)]
        native = part.loc[part.native224_20frame_eligible]
        family_rows.append({"family": family, "distinct_analytes": len(group), "members": _join_unique(group.canonical_name),
                            "any_member_matched24h_videos": part.video_id.nunique(), "any_member_matched24h_patients": part.hospital_id.nunique(),
                            "any_member_native224_20frame_videos": native.video_id.nunique(), "not_a_pooled_prediction_target": True})
    pd.DataFrame(family_rows).sort_values("any_member_matched24h_videos", ascending=False).to_csv(TABLES / "family_coverage.csv", index=False, encoding="utf-8-sig")
    _figure_style()
    figures = HERE / "outputs/figures/video24h_matching"
    figures.mkdir(parents=True, exist_ok=True)
    positive = canonical.loc[canonical.matched24h_videos.gt(0)]
    for page, start in enumerate(range(0, len(positive), 25), 1):
        frame = positive.iloc[start:start + 25].iloc[::-1]
        figure, axis = plt.subplots(figsize=(15, max(5, len(frame) * .38)))
        y = np.arange(len(frame))
        raw_bars = axis.barh(y + .18, frame.matched24h_videos, .35, color="#64737C", label="24h matched sessions")
        axis.bar_label(raw_bars, padding=3, fontsize=7)
        bars = axis.barh(y - .18, frame.native224_20frame_videos, .35, color="#2878B5", label="Native224 / 20-frame eligible")
        axis.bar_label(bars, padding=3, fontsize=7)
        labels = [f"{row.canonical_name} [{row.canonical_unit}] | {row.specimen_group}" for row in frame.itertuples()]
        axis.set_yticks(y, labels, fontsize=8)
        axis.set(xlabel="Unique videos", title=f"Same-admission nearest exact lab result within 24h | ranks {start+1}-{start+len(frame)}")
        axis.legend(); axis.grid(axis="x", alpha=.2); axis.margins(x=.1); figure.tight_layout()
        figure.savefig(figures / f"ranking_{page:02d}.png", dpi=180, bbox_inches="tight"); plt.close(figure)
    reports = HERE / "outputs/reports"; reports.mkdir(parents=True, exist_ok=True)
    lines = ["# 全量化验项目与视频24h匹配统计", "", f"当前化验总表：{len(labs):,}行，{labs['检验项名称'].nunique()}个原始项目名；{len(source)}种名称/单位/标本字段，归为{len(canonical)}个可区分项目。",
             f"视频口径：{len(videos)}个已验证同次住院session，{videos.hospital_id.nunique()}名患者；其中{int(videos.native224_20frame_eligible.sum())}个视频满足当前224协议和20非相邻帧条件。", "",
             "## 统计口径", "", "- 仅使用当前merged_lab_tests.csv；分散CSV、转换表和备份不重复叠加。来源清单见source_table_inventory.csv。",
             "- 化验、入院、出院采用上海时区；视频以Session Timestamp为起点，原始帧时间仅用于时长。匹配同患者、同次住院内的最近报告。",
             "- 24h为视频原始采集区间到化验时点的最小距离；区间内部距离为0，边界包含24h。并列时按距视频中点、报告时间排序。",
             "- 排序列matched24h_videos使用精确、有限的数值报告；比较符、范围、定性或序数结果不当作精确连续值。any_result列额外统计非空结果的时间覆盖。",
             "- 每视频每项目最多计一次；同时间多行或等价别名取中位数。不同项目计数不可相加当作总视频数。",
             "- 复用现有明确别名及单位换算；按标本背景区分血/尿/其他，血气项目的动脉、静脉、未注明背景分别保留。",
             "- PO2与体温校正PO2、实际与标准化P50、总/直接/间接胆红素、肌酐法/胱抑素C法eGFR、BNP/NT-proBNP、肌钙蛋白I/T等不互相混合。",
             "- family只用于组织同类指标，不意味着可互换；family_coverage的覆盖数是视频并集，不是一个合并预测标签。检测套名称和原字段映射均保留。",
             "- 此表是全项目可用性统计，未为每个新项目猜测生理范围或临床阈值，因此不等于直接可训练样本数；已有Exp2任务的专属范围/标本规则仍以训练协议为准。",
             "- 原化验行缺住院信息时不丢弃数值：由该患者总表中可验证的住院记录约束视频和报告匹配。无可验证住院episode的视频不纳入。",
             "- 本次独立增量统计不重跑或覆盖此前纵向轨迹结果；旧轨迹的数据时效性记录在本次manifest中。", "", "## 降序列表", "",
             "|排名|项目|单位|标本背景|24h视频|患者|224/20帧视频|对应独立化验时点|同类|", "|---:|---|---|---|---:|---:|---:|---:|---|"]
    for row in canonical.itertuples():
        escape = lambda value: str(value).replace("|", "\\|")
        lines.append(f"|{row.rank}|{escape(row.canonical_name)}|{escape(row.canonical_unit)}|{row.specimen_group}|{row.matched24h_videos}|{row.matched24h_patients}|{row.native224_20frame_videos}|{row.matched24h_distinct_lab_events}|{row.family}|")
    lines.extend(["", "完整原始字段排名、合并映射、逐视频匹配、排除原因与来源清单：`../tables/video24h_matching/`。图：`../figures/video24h_matching/ranking_*.png`。"])
    (reports / "VIDEO24H_MATCHING.md").write_text("\n".join(lines) + "\n")
    metadata_dir = HERE / "outputs/metadata"; metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "video24h_matching_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main():
    digest = _sha256(LAB)
    labs, episodes, audit = load_labs()
    print(f"[labs-ready] {len(labs)} rows / {labs.canonical_id.nunique()} distinct canonical+specimen groups", flush=True)
    videos, statuses = video_inventory(episodes)
    videos, reused = native_frame_eligibility(videos)
    matches = match_events(labs, videos, "canonical_id")
    print(f"[matched] canonical exact values: {len(matches)} video/analyte pairs", flush=True)
    raw_matches = match_events(labs, videos, "source_id")
    print(f"[matched] raw fields: {len(raw_matches)} video/field pairs", flush=True)
    any_matches = match_events(labs, videos, "canonical_id", numeric=False)
    raw_any = match_events(labs, videos, "source_id", numeric=False)
    canonical = ranked_catalog(labs, matches, any_matches, "canonical_id")
    source = ranked_catalog(labs, raw_matches, raw_any, "source_id")
    if matches.duplicated(["canonical_id", "video_id"]).any() or raw_matches.duplicated(["source_id", "video_id"]).any():
        raise RuntimeError("A video was counted more than once within a target")
    old_quality = json.loads((HERE / "outputs/metadata/data_quality_report.json").read_text())
    metadata = {"lab_table": str(LAB), "lab_table_sha256": digest, "lab_rows": len(labs), "raw_item_names": labs["检验项名称"].nunique(),
                "source_fields": len(source), "canonical_specimen_groups": len(canonical), "matching_hours": 24,
                "same_admission": True, "video_time_source": "validated Session Timestamp plus original capture duration",
                "timezone": "Asia/Shanghai", "source_video_status_counts": statuses, "video_sessions": len(videos),
                "video_patients": videos.hospital_id.nunique(), "native224_20frame_videos": int(videos.native224_20frame_eligible.sum()),
                "matching_unit": "one nearest report per video/field; interval distance then midpoint distance then earlier timestamp",
                "numeric_results": "finite exact full-string numbers, valid thousands separators, no censored/range/ordinal extraction",
                "lab_row_episode_metadata_required": False, "blood_gas_specimens_separate": True,
                "physiological_range_filter": "none in descriptive rankings; existing task-specific training policies are not changed",
                "canonical_lab_events": labs.loc[labs.numeric_valid, ["canonical_id", "hospital_id", "report_unix"]].drop_duplicates().shape[0],
                "reused_frame_indexes": reused, "new_pixel_or_video_caches": False,
                "existing_longitudinal_source_sha256": old_quality["source"]["sha256"],
                "existing_longitudinal_results_use_current_table": old_quality["source"]["sha256"] == digest,
                "analysis_code_sha256": _sha256(__file__), "harmonization_code_sha256": _sha256(HERE / "run_analysis.py"),
                "video_audit_sha256": _sha256(REFERENCE / "source_data/raw_video_audit.csv"),
                "no_training": True, "outputs": "tables/video24h_matching; reports/VIDEO24H_MATCHING.md; figures/video24h_matching"}
    if _sha256(LAB) != digest:
        raise RuntimeError("Lab table changed during the audit")
    write_outputs(labs, videos, canonical, source, matches, raw_matches, audit, metadata)
    print(canonical.head(25)[["rank", "canonical_name", "canonical_unit", "specimen_group", "matched24h_videos", "matched24h_patients", "native224_20frame_videos"]].to_string(index=False), flush=True)
    print(f"[complete] {HERE / 'outputs/reports/VIDEO24H_MATCHING.md'}", flush=True)


if __name__ == "__main__":
    main()
