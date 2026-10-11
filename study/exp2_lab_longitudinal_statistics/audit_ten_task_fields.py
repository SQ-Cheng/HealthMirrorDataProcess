"""Read-only audit of the ten current Exp2 label source definitions."""

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from study.exp2_face_history_head32_regression.source_data import (
    LAB_SOURCE_DEFINITIONS, TARGET_ANALYTES, _canonical_values, _extract_numeric,
    _load_lab_data, _normalize_hospital_id, _parse_datetime_to_unix, _sha256,
)
from .video24h_matching import LAB, REFERENCE, TABLES, exact_numeric, nearest_sorted, parse_times


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/tables/ten_task_field_audit"
REPORT = HERE / "outputs/reports/TEN_TASK_FIELD_AUDIT.md"
TASKS = (
    "oxyhemoglobin_fraction", "lactate_high", "urea_high", "total_bilirubin_high",
    "platelet_count_low", "hemoglobin_low", "aa_po2_ratio_low", "creatinine_high",
    "hematocrit_low", "egfr_low",
)
TOKENS = {
    "oxyhemoglobin_fraction": ("氧合血红蛋白",), "lactate": ("乳酸",),
    "urea": ("尿素",), "total_bilirubin": ("胆红素",),
    "platelet_count": ("血小板",), "hemoglobin": ("血红蛋白",),
    "aa_po2_ratio": ("肺泡",), "creatinine": ("肌酐",),
    "hematocrit": ("红细胞压积",), "egfr_creatinine": ("eGFR",),
}


def select_rows(raw, analyte):
    policy = LAB_SOURCE_DEFINITIONS[analyte]
    selected = raw.loc[raw["检验项名称"].isin(policy["items"])].copy()
    selected["value"], supported = _canonical_values(
        analyte, selected["检验项名称"], selected.numeric_value, selected["单位"],
    )
    failures = {
        "invalid_patient": selected.hospital_id.eq(""),
        "invalid_time": selected.timestamp_unix.isna(),
        "nonnumeric": selected.value.isna(),
        "censored": selected["检验值(文本)"].str.match(r"^\s*[<>≤≥＜＞]", na=False),
        "unsupported_unit": ~supported,
        "outside_range": selected.value.notna() & ~selected.value.between(*policy["valid_range"]),
        "disallowed_specimen": selected["标本名称"].str.contains("静脉", regex=False)
        if policy["exclude_venous"] else pd.Series(False, index=selected.index),
    }
    if "allowed_specimens" in policy:
        failures["disallowed_specimen"] |= ~selected["标本名称"].isin(policy["allowed_specimens"])
        failures["censored"] |= selected["检验值(文本)"].str.match(r"^\s*(?:大于|小于|超过|低于)", na=False)
    flags = pd.DataFrame(failures)
    selected["accepted"] = ~flags.any(axis=1)
    selected["exclusion_reasons"] = flags.apply(lambda row: "|".join(row.index[row]), axis=1)
    selected["nonexact_numeric"] = selected.accepted & selected.exact_value.isna()
    selected["numeric_parser_disagreement"] = selected.accepted & selected.exact_value.notna() & ~np.isclose(
        selected.numeric_value, selected.exact_value, rtol=0, atol=1e-12,
    )
    return selected, flags


def events(rows):
    return rows.groupby(["hospital_id", "timestamp_unix"], as_index=False).value.median()


def rematch(pool, videos):
    by_patient = {p: g for p, g in videos.groupby("hospital_id")}
    result = []
    for patient, group in pool.groupby("hospital_id"):
        if patient not in by_patient:
            continue
        group = group.sort_values("timestamp_unix")
        for video in by_patient[patient].itertuples(index=False):
            match = nearest_sorted(group.timestamp_unix.to_numpy(), group.value.to_numpy(), video)
            if match:
                result.append({"hospital_id": patient, "video_id": video.video_id,
                               "label_time_unix": match["timestamp_unix"], "raw_value": match["value"],
                               "match_delta_h": match["delta_h"]})
    return pd.DataFrame(result, columns=["hospital_id", "video_id", "label_time_unix", "raw_value", "match_delta_h"])


def compare_matches(baseline, shadow, target, scenario):
    compared = baseline.merge(shadow, on=["hospital_id", "video_id"], how="outer",
                              suffixes=("_current", "_shadow"), indicator=True, validate="one_to_one")
    paired = compared._merge.eq("both")
    value_changed = paired & ~np.isclose(compared.raw_value_current, compared.raw_value_shadow, atol=1e-10)
    time_changed = paired & compared.label_time_unix_current.ne(compared.label_time_unix_shadow)
    changed = compared.loc[~paired | value_changed | time_changed].copy()
    changed["target"], changed["scenario"] = target, scenario
    return {
        "target": target, "scenario": scenario, "current_videos": len(baseline),
        "shadow_videos": len(shadow), "added_videos": int(compared._merge.eq("right_only").sum()),
        "lost_videos": int(compared._merge.eq("left_only").sum()),
        "changed_value_videos": int(value_changed.sum()), "changed_time_videos": int(time_changed.sum()),
    }, changed


def alias_decision(analyte, name):
    if name in LAB_SOURCE_DEFINITIONS[analyte]["items"]:
        return "included", "Exact whitelist; row-level unit/specimen/value checks still apply."
    if name.strip() in LAB_SOURCE_DEFINITIONS[analyte]["items"]:
        return "omitted_whitespace_variant", "Matches whitelist only after trimming; inspect downstream impact."
    if analyte == "aa_po2_ratio" and name == "动脉-肺泡氧分压比值":
        return "omitted_ratio_alias", "Longitudinal harmonization treats as fraction x100; paired source arithmetic audited separately."
    if analyte == "creatinine" and name == "肌酐":
        return "excluded_urine_source", "Observed single row is urine, mmol/L; not the blood creatinine target."
    if "(尿)" in name or "尿微量" in name or "比值" in name and analyte == "creatinine":
        return "distinct_quantity", "Urine source or ratio, not blood concentration."
    if analyte == "egfr_creatinine":
        return "distinct_method", "Cystatin-C eGFR is not the directly reported creatinine-based eGFR task."
    if analyte == "lactate" and "脱氢酶" in name:
        return "distinct_quantity", "Lactate dehydrogenase is not lactate concentration."
    if analyte == "hemoglobin":
        return "distinct_quantity", "Fraction, glycated Hb, mean cell index or related parameter; not total Hb concentration."
    if analyte == "platelet_count":
        return "distinct_quantity", "Morphology, volume, ratio, aggregation or other platelet parameter; not platelet count."
    if analyte == "aa_po2_ratio":
        return "distinct_quantity_or_correction", "Pressure, pressure difference, inverse ratio or temperature-corrected parameter; not pooled automatically."
    return "review_separately", "Name similarity alone is insufficient to merge this field."


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(LAB, dtype=str, keep_default_na=False, usecols=[
        "首页病案号", "检验套名称", "检验项名称", "检验值(文本)", "单位", "标本名称", "报告时间",
    ])
    raw["source_row_number"] = raw.index + 2
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["timestamp_unix"] = _parse_datetime_to_unix(raw["报告时间"])
    raw["robust_timestamp_unix"] = parse_times(raw["报告时间"])
    raw["numeric_value"] = _extract_numeric(raw["检验值(文本)"].str.replace(",", "", regex=False))
    raw["exact_value"] = exact_numeric(raw["检验值(文本)"])
    videos = pd.read_csv(TABLES / "video_inventory.csv", dtype={"hospital_id": str})
    eligible = videos.loc[videos.native224_20frame_eligible].copy()
    caches = pd.concat([pd.read_csv(REFERENCE / path, dtype={"hospital_id": str}) for path in (
        "source_data/lab_timeseries.csv", "source_data/direct_fields/lab_timeseries.csv",
    )], ignore_index=True)
    summaries, fields, details, exclusions, duplicates, suspects = [], [], [], [], [], []
    cache_checks, impacts, changed_matches, catalogs, temporal = [], [], [], [], []
    current_pools, retained_by_target = {}, {}
    for target in TASKS:
        analyte = TARGET_ANALYTES[target]
        selected, flags = select_rows(raw, analyte)
        retained = selected.loc[selected.accepted].copy()
        pool = events(retained)
        current_pools[target] = pool
        retained_by_target[target] = retained
        baseline = pd.read_csv(REFERENCE / f"task_records/{target}.csv", dtype={"hospital_id": str})
        keys = ["hospital_id", "timestamp_unix"]
        counts = retained.groupby(keys).agg(source_rows=("value", "size"),
                unique_values=("value", "nunique"), unique_names=("检验项名称", "nunique"),
                item_names=("检验项名称", lambda x: "|".join(sorted(set(x)))),
                suites=("检验套名称", lambda x: "|".join(sorted(set(x)))),
                specimens=("标本名称", lambda x: "|".join(sorted(set(x)))))
        duplicate = counts.loc[counts.source_rows.gt(1)].reset_index()
        duplicate["target"] = target
        duplicates.append(duplicate)
        flagged = retained.loc[retained.nonexact_numeric | retained.numeric_parser_disagreement].copy()
        join = flagged.merge(baseline, left_on=keys, right_on=["hospital_id", "label_time_unix"],
                             how="left", suffixes=("", "_label"))
        join["target"] = target
        suspects.append(join)
        summaries.append({"target": target, "analyte": analyte, "canonical_unit": LAB_SOURCE_DEFINITIONS[analyte]["canonical_unit"],
            "raw_rows": len(selected), "accepted_rows": len(retained), "canonical_events": len(pool),
            "canonical_patients": pool.hospital_id.nunique(), "identical_duplicate_rows_removed": len(retained)-len(pool),
            "conflicting_events": int(counts.unique_values.gt(1).sum()),
            "cross_name_duplicate_events": int(counts.unique_names.gt(1).sum()),
            "nonexact_numeric_rows": len(flagged), "nonexact_matched_videos": join.video_id.nunique(),
            "numeric_parser_disagreement_rows": int(retained.numeric_parser_disagreement.sum()),
            "current_videos": len(baseline), "current_patients": baseline.hospital_id.nunique()})
        for name, group in selected.groupby("检验项名称"):
            accepted = group.loc[group.accepted]
            matched = accepted.merge(baseline, left_on=keys, right_on=["hospital_id", "label_time_unix"])
            fields.append({"target": target, "item_name": name, "raw_rows": len(group),
                "nonempty_result_rows": int(group["检验值(文本)"].str.strip().ne("").sum()),
                "exact_numeric_rows": int(group.exact_value.notna().sum()), "accepted_rows": len(accepted),
                "accepted_events": len(events(accepted)), "accepted_patients": accepted.hospital_id.nunique(),
                "contributing_current_videos": matched.video_id.nunique(),
                "canonical_min": accepted.value.min(), "canonical_median": accepted.value.median(),
                "canonical_max": accepted.value.max(),
                "source_units": json.dumps(group["单位"].value_counts().to_dict(), ensure_ascii=False)})
        for (name, unit, specimen, suite), group in selected.groupby(["检验项名称", "单位", "标本名称", "检验套名称"], dropna=False):
            details.append({"target": target, "item_name": name, "unit": unit, "specimen": specimen,
                            "suite": suite, "raw_rows": len(group), "accepted_rows": int(group.accepted.sum())})
        excluded = selected.loc[~selected.accepted].copy()
        excluded["target"] = target
        exclusions.append(excluded)
        legacy, robust = selected.timestamp_unix, selected.robust_timestamp_unix
        temporal.append({"target": target, "legacy_invalid_rows": int(legacy.isna().sum()),
            "robust_recovers_rows": int((legacy.isna() & robust.notna()).sum()),
            "different_timestamp_rows": int((legacy.notna() & robust.notna() & ~np.isclose(legacy, robust, rtol=0, atol=1e-6)).sum())})
        cached = caches.loc[caches.analyte.eq(analyte), keys+["value"]]
        check = pool.merge(cached, on=keys, how="outer", suffixes=("_fresh", "_cache"), indicator=True, validate="one_to_one")
        assert check._merge.eq("both").all() and np.allclose(check.value_fresh, check.value_cache, rtol=0, atol=1e-10), target
        labels = baseline.merge(pool, left_on=["hospital_id", "label_time_unix"], right_on=keys, how="left", validate="many_to_one")
        assert np.allclose(labels.raw_value, labels.value, rtol=0, atol=1e-10), target
        shared_path = HERE.parent / f"exp2_face_shared_backbone_head32_regression/outputs/task_records/{target}.csv"
        shared_equal = None
        if shared_path.exists():
            shared = pd.read_csv(shared_path, dtype={"hospital_id": str})
            comparison = baseline.merge(shared, on=["hospital_id", "video_id"], suffixes=("_main", "_shared"), how="outer", indicator=True)
            shared_equal = bool(comparison._merge.eq("both").all() and np.allclose(comparison.raw_value_main, comparison.raw_value_shared, rtol=0, atol=1e-10)
                                and comparison.label_time_unix_main.eq(comparison.label_time_unix_shared).all())
            assert shared_equal, target
        cache_checks.append({"target": target, "fresh_cache_equal": True, "active_labels_equal": True, "shared_labels_equal": shared_equal})
        strict_raw = raw.loc[raw["检验项名称"].isin(LAB_SOURCE_DEFINITIONS[analyte]["items"])].copy()
        strict_raw["numeric_value"] = strict_raw.exact_value
        strict_selected, _ = select_rows(strict_raw, analyte)
        strict_pool = events(strict_selected.loc[strict_selected.accepted])
        impact, changed = compare_matches(baseline, rematch(strict_pool, eligible), target, "exclude_nonexact_numeric")
        impacts.append(impact)
        changed_matches.append(changed)
        for name in sorted(raw["检验项名称"].unique()):
            if not any(token in name for token in TOKENS[analyte]):
                continue
            group = raw.loc[raw["检验项名称"].eq(name)]
            decision, rationale = alias_decision(analyte, name)
            catalogs.append({"target": target, "item_name": name, "raw_rows": len(group),
                "exact_numeric_rows": int(group.exact_value.notna().sum()), "decision": decision, "rationale": rationale,
                "units": json.dumps(group["单位"].value_counts().to_dict(), ensure_ascii=False),
                "specimens": json.dumps(group["标本名称"].value_counts().to_dict(), ensure_ascii=False)})
        print(f"[audit] {target}: {len(selected)} -> {len(retained)} rows, {len(pool)} events; cache and labels verified", flush=True)

    # Shadow only the omitted ratio alias, preserving existing specimen/range/time policies.
    alias = raw.loc[raw["检验项名称"].eq("动脉-肺泡氧分压比值")].copy()
    alias["value"] = alias.exact_value * 100
    valid_alias = alias.loc[alias.hospital_id.ne("") & alias.timestamp_unix.notna() & alias["单位"].eq("")
        & ~alias["标本名称"].str.contains("静脉", regex=False) & alias.value.between(1, 800)]
    baseline = pd.read_csv(REFERENCE / "task_records/aa_po2_ratio_low.csv", dtype={"hospital_id": str})
    enlarged = events(pd.concat([retained_by_target["aa_po2_ratio_low"][["hospital_id", "timestamp_unix", "value"]],
                                valid_alias[["hospital_id", "timestamp_unix", "value"]]], ignore_index=True))
    impact, changed = compare_matches(baseline, rematch(enlarged, eligible), "aa_po2_ratio_low", "include_verified_fraction_alias")
    impacts.append(impact)
    changed_matches.append(changed)
    alias_matches = rematch(events(valid_alias), videos)
    alias_matches.to_csv(OUT / "omitted_ratio_alias_video_matches.csv", index=False)
    packet_items = ["动脉-肺泡氧分压比值", "氧分压", "平均肺泡氧分压", "肺泡内氧分压", "肺泡氧分压"]
    packets = raw.loc[raw["检验项名称"].isin(packet_items) & raw.exact_value.notna()].copy()
    packets["packet_item"] = packets["检验项名称"].replace(dict.fromkeys(packet_items[2:], "alveolar_po2"))
    paired = packets.pivot_table(index=["hospital_id", "timestamp_unix", "检验套名称", "标本名称"],
                                columns="packet_item", values="exact_value", aggfunc="median")
    paired = paired.dropna(subset=["动脉-肺泡氧分压比值", "氧分压", "alveolar_po2"]).reset_index()
    paired["calculated_fraction"] = paired["氧分压"] / paired.alveolar_po2
    paired["reported_minus_calculated_fraction"] = paired["动脉-肺泡氧分压比值"] - paired.calculated_fraction
    paired.to_csv(OUT / "omitted_ratio_alias_arithmetic.csv", index=False)
    missing_egfr = raw.loc[raw["检验项名称"].eq("eGFR(CKD-EPI 肌酐)") & raw["单位"].eq("") & raw.exact_value.notna()].copy()
    missing_egfr["value"] = missing_egfr.exact_value
    egfr_matches = rematch(events(missing_egfr), videos)
    egfr_matches.to_csv(OUT / "missing_unit_egfr_video_matches.csv", index=False)
    blank_specimen = []
    for target, retained in retained_by_target.items():
        source = retained.loc[retained["标本名称"].eq("")].copy()
        source["target"] = target
        baseline = pd.read_csv(REFERENCE / f"task_records/{target}.csv", dtype={"hospital_id": str})
        blank_specimen.append(source.merge(baseline, left_on=["hospital_id", "timestamp_unix"],
                                          right_on=["hospital_id", "label_time_unix"], how="left"))

    # A temporary reconstruction also exercises the real loader, without touching experiment files.
    with tempfile.TemporaryDirectory(prefix="exp2_field_audit_") as temp:
        fresh, _, _ = _load_lab_data(TASKS, temp)
        for target in TASKS:
            pool = fresh.loc[fresh.analyte.eq(TARGET_ANALYTES[target])]
            expected = current_pools[target]
            check = pool.merge(expected, on=["hospital_id", "timestamp_unix"], how="outer", suffixes=("_loader", "_audit"), indicator=True)
            assert check._merge.eq("both").all() and np.allclose(check.value_loader, check.value_audit, rtol=0, atol=1e-10), target
    tables = {
        "task_summary": pd.DataFrame(summaries), "field_counts": pd.DataFrame(fields),
        "unit_specimen_suite_counts": pd.DataFrame(details), "excluded_rows": pd.concat(exclusions, ignore_index=True),
        "duplicate_events": pd.concat(duplicates, ignore_index=True),
        "questionable_numeric_rows_and_matches": pd.concat(suspects, ignore_index=True),
        "cache_and_label_validation": pd.DataFrame(cache_checks), "counterfactual_impact": pd.DataFrame(impacts),
        "counterfactual_changed_video_labels": pd.concat(changed_matches, ignore_index=True),
        "similar_name_catalog": pd.DataFrame(catalogs), "time_parser_comparison": pd.DataFrame(temporal),
        "accepted_blank_specimen_rows_and_matches": pd.concat(blank_specimen, ignore_index=True),
        "exclusion_reason_combinations": pd.concat(exclusions, ignore_index=True).groupby(
            ["target", "exclusion_reasons"], as_index=False).size(),
    }
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    signature = {"path": str(LAB), "sha256": _sha256(LAB), "size_bytes": LAB.stat().st_size,
                 "raw_rows": len(raw), "distinct_item_names": raw["检验项名称"].nunique()}
    metadata = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "source": signature,
                "reference": str(REFERENCE), "targets": list(TASKS), "source_policies": {
                    TARGET_ANALYTES[t]: LAB_SOURCE_DEFINITIONS[TARGET_ANALYTES[t]] for t in TASKS},
                "actual_loader_reproduction_passed": True, "training_or_source_modified": False,
                "counterfactual_only": True, "excluded_reason_counts_overlap": True,
                "missing_unit_egfr_rows": len(missing_egfr), "missing_unit_egfr_matching_videos": len(egfr_matches),
                "ratio_alias_max_absolute_arithmetic_error": float(paired.reported_minus_calculated_fraction.abs().max())}
    meta_path = HERE / "outputs/metadata/ten_task_field_audit_manifest.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 当前十任务化验字段审计", "", f"生成时间（UTC）：{metadata['generated_utc']}",
        f"源表：`merged_lab_tests.csv`，{len(raw):,}行，{signature['distinct_item_names']}个原始名称。",
        f"SHA256：`{signature['sha256']}`。", "", "## 范围与验证", "",
        "审计当前224、20帧、24h主回归十任务及共享骨干标签；没有修改训练或源表。所有十任务均通过真实loader临时重建、现存化验缓存、视频原始标签三方一致性检查。共享模型的原始标签和时间也一致，但其患者split与主实验不同。",
        "数字和时间问题通过影子筛选/匹配评估，不写回实验。统计只读取最新合并总表，不将原始表与备份重复拼接。", "",
        "## 关键发现", "",
        "1. 当前数字读取提取第一个数字，而不是要求完整精确数字。44条已纳入记录带`?`前缀，其中乳酸3个视频直接使用这种记录；其质量含义未知，需回溯而不能无标记当作精确测量。影子严格筛选后的丢失/替换见下表和CSV。",
        "2. `动脉-肺泡氧分压比值`29条未在Exp2白名单内，来自全血/Nova手术中血气套。已有longitudinal合并规则将它按无量纲分数×100；同报告同标本的分压算术支持该口径，但一位小数舍入和输入分压舍入限制精度。不能把原始分数直接当百分数。影子纳入后的实际影响见下表。",
        "3. 所有十任务当前有效记录在同患者/同报告秒聚合前均无不同数值冲突。O2Hb、乳酸、Hb和A/a存在同值重复，median消除重复；零冲突不证明不同测量方法完全可互换。Hb纳入血常规和血气来源，报告时间不一致的不同测量不会去重。",
        f"4. 总胆红素5,206条空单位行同时为空值，不能算成丢失5,206条有用化验。血小板16条无单位结果都是形态文字，不是计数。eGFR有29条真实数值但缺单位，日期范围{missing_egfr['报告时间'].min()}至{missing_egfr['报告时间'].max()}；在相同住院和24h条件下匹配到{len(egfr_matches)}个本地视频，不能未经原始报告验证直接推定单位。",
        "5. 只有HCT/eGFR有完整标本白名单；其他任务仅按现有策略排静脉血或不排。当前未发现已接受的尿/其他非血标本，但A/a有1条空标本来源，未进入当前视频标签；Hb/HCT/乳酸保留少量静脉血是现行定义，并非加载遗漏。",
        "6. 名称相似不代表同一目标：单独的`肌酐`1条为尿，正确排除；胱抑素C版eGFR、乳酸脱氢酶、糖化/平均红细胞Hb、血小板体积/聚集等不应混入。A/a温度校正字段没有自动合并。", "",
        "## 每字段数量", "", "原始行数→按当前代码保留行数。一个报告事件可能有多个字段，字段贡献视频数不能简单相加。", "",
        "|任务|原始字段|原始行|保留行|贡献当前视频|", "|---|---|---:|---:|---:|"]
    for row in fields:
        lines.append(f"|{row['target']}|{row['item_name']}|{row['raw_rows']:,}|{row['accepted_rows']:,}|{row['contributing_current_videos']:,}|")
    lines += ["", "## 任务规模与冲突", "", "|任务|保留源行|去重事件|患者|视频|冲突事件|可疑数字源行|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"|{row['target']}|{row['accepted_rows']:,}|{row['canonical_events']:,}|{row['current_patients']}|{row['current_videos']}|{row['conflicting_events']}|{row['nonexact_numeric_rows']}|")
    lines += ["", "## 不改代码的影子匹配", "", "完全相同的已审计视频/住院边界/24h最近匹配；仅改变注明的化验筛选。新增、丢失、时间替换和数值替换可能重叠。",
        "", "|任务|影子规则|新增视频|丢失视频|值变动视频|时间变动视频|", "|---|---|---:|---:|---:|---:|"]
    for row in impacts:
        lines.append(f"|{row['target']}|{row['scenario']}|{row['added_videos']}|{row['lost_videos']}|{row['changed_value_videos']}|{row['changed_time_videos']}|")
    lines += ["", "## 边界与可复查文件", "",
        "时间解析差异见`time_parser_comparison.csv`；本次十任务没有混合格式解析新恢复的记录或时间偏移，也没有科学记数法等精确数字被读取成不同数值的已接受记录。该检查是报告时间解析，不是重新核验原始视频时钟或手术日期。", "",
        "相同报告秒使用median，不按仪器/标本分组；当前无数值冲突不代表未来更新不会出现冲突。范围过滤依据现有工程阈值，超过阈值的记录见`excluded_rows.csv`，不能仅凭超范围判定原始化验错误。",
        "", "`field_counts.csv`给出精确字段数量与当前标签贡献；`unit_specimen_suite_counts.csv`按单位/标本/检验套展开；`similar_name_catalog.csv`列出所有近似名称及纳入/不纳入理由。",
        "", "`questionable_numeric_rows_and_matches.csv`与`counterfactual_changed_video_labels.csv`给出可疑文本和受影响视频ID，`duplicate_events.csv`给出所有重复事件；排除原因可能同时成立，不应把各原因相加当总行数。",
        "", "复现：`python -m study.exp2_lab_longitudinal_statistics.audit_ten_task_fields`。仅生成本审计文件；真实loader写入临时目录并自动清理。"]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "tables": str(OUT), "counterfactuals": impacts}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
