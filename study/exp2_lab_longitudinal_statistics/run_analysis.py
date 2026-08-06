"""Analyze patient-balanced laboratory trajectories during hospitalization."""

from collections import Counter
import argparse
import hashlib
import json
import math
import os
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import rankdata, theilslopes, wilcoxon

from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id
from study.exp2_lab_multimodal.config import LAB_CSV


EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, "outputs")
MIN_REPEATED_EPISODES = 20
MIN_PAIRED_PATIENTS = 20
PROGRESS_BIN_COUNT = 10
BOOTSTRAP_REPLICATES = 2000
SEED = 20260730
MG_DL_PER_MMOL_L_GLUCOSE = 18.0182

REQUIRED_COLUMNS = (
    "首页病案号",
    "首页性别",
    "首页就诊时年龄",
    "首页入院时间",
    "首页出院时间",
    "首页住院天数",
    "手术开始日期",
    "手术结束日期",
    "首页手术操作名称",
    "检验套名称",
    "检验项名称",
    "检验值(文本)",
    "单位",
    "标本名称",
    "报告时间",
)

ENGLISH_NAMES = {
    "*平均血红蛋白浓度": "Mean corpuscular Hb concentration",
    "*平均血红蛋白量": "Mean corpuscular hemoglobin",
    "*糖化血红蛋白": "HbA1c",
    "*肌钙蛋白Ⅰ(hsTnI)测定": "High-sensitivity troponin I",
    "*葡萄糖(Glu)测定": "Laboratory glucose",
    "*血红蛋白": "Laboratory hemoglobin",
    "乳酸浓度": "Lactate",
    "二氧化碳分压": "PCO2",
    "动脉氧分压与肺泡氧分压之比": "Arterial/alveolar PO2 ratio",
    "动脉血氧分压与肺泡内氧分压之比": "Arterial/alveolar oxygen ratio",
    "在PH7.4,PCO2=40mmHg,体温37度SO2=50%的氧分压": "Standardized P50",
    "平均肺泡氧分压": "Mean alveolar PO2",
    "总血红蛋白": "Total hemoglobin",
    "患者体温下二氧化碳分压": "Temperature-corrected PCO2",
    "患者体温下动脉氧分压与肺泡氧分压之比": "Temperature-corrected PaO2/PAO2 ratio",
    "患者体温下平均肺泡氧分压": "Temperature-corrected mean alveolar PO2",
    "患者体温下氧分压": "Temperature-corrected PO2",
    "患者体温下氧饱和度50%时的氧分压": "Patient-condition P50",
    "患者体温下肺泡动脉氧分压差": "Temperature-corrected A-a PO2 gradient",
    "标准状态下氧饱和度50%时的氧分压": "Standard-condition P50",
    "氧分压": "PO2",
    "氧合血红蛋白": "Oxyhemoglobin",
    "氧合血红蛋白分数": "Oxyhemoglobin fraction",
    "氧饱和度50%时的氧分压": "P50",
    "氧饱和度50%的氧分压": "P50 (alternate)",
    "碳氧血红蛋白": "Carboxyhemoglobin",
    "碳氧血红蛋白分数": "Carboxyhemoglobin fraction",
    "糖化血红蛋白": "HbA1c",
    "肌钙蛋白Ⅰ(hsTnI)测定": "Troponin I",
    "肺泡内氧分压": "Alveolar PO2",
    "肺泡内氧分压与动脉血氧分压之差": "Alveolar-arterial PO2 difference",
    "肺泡动脉氧分压差": "A-a PO2 gradient",
    "葡萄糖浓度": "Blood-gas glucose",
    "血氧浓度50%氧分压": "P50 from oxygen concentration",
    "血液中氧分压与吸氧浓度之比": "P/F ratio",
    "血红蛋白": "Blood-gas hemoglobin",
    "还原血红蛋白": "Deoxyhemoglobin",
    "还原血红蛋白分数": "Deoxyhemoglobin fraction",
    "高铁血红蛋白": "Methemoglobin",
    "高铁血红蛋白分数": "Methemoglobin fraction",
}

HARMONIZATION_RULES = (
    {
        "rule_id": "blood_gas_glucose_mgdl_to_mmoll",
        "source_item": "葡萄糖浓度",
        "source_units": ("mg/dl",),
        "canonical_item": "葡萄糖浓度",
        "canonical_unit": "mmol/l",
        "factor": 1.0 / MG_DL_PER_MMOL_L_GLUCOSE,
        "formula": "mmol/L = mg/dL / 18.0182",
        "rationale": "Same blood-gas glucose analyte expressed in mass units.",
    },
    {
        "rule_id": "blood_gas_glucose_mmoll_identity",
        "source_item": "葡萄糖浓度",
        "source_units": ("mmol/l",),
        "canonical_item": "葡萄糖浓度",
        "canonical_unit": "mmol/l",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Canonical blood-gas glucose unit.",
    },
    {
        "rule_id": "arterial_alveolar_ratio_percent_identity",
        "source_item": "动脉氧分压与肺泡氧分压之比",
        "source_units": ("%",),
        "canonical_item": "动脉氧分压与肺泡氧分压之比",
        "canonical_unit": "%",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Canonical percentage representation of PaO2/PAO2.",
    },
    {
        "rule_id": "arterial_alveolar_ratio_fraction_to_percent",
        "source_item": "动脉血氧分压与肺泡内氧分压之比",
        "source_units": ("unitless",),
        "canonical_item": "动脉氧分压与肺泡氧分压之比",
        "canonical_unit": "%",
        "factor": 100.0,
        "formula": "percent = fraction * 100",
        "rationale": "Direct formula audit confirms PO2/alveolar PO2 fraction.",
    },
    {
        "rule_id": "standard_condition_p50_explicit_alias",
        "source_item": "在PH7.4,PCO2=40mmHg,体温37度SO2=50%的氧分压",
        "source_units": ("mmhg", "unitless"),
        "canonical_item": "标准状态下氧饱和度50%时的氧分压",
        "canonical_unit": "mmhg",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Explicit definition of P50 under standard conditions.",
    },
    {
        "rule_id": "standard_condition_p50_identity",
        "source_item": "标准状态下氧饱和度50%时的氧分压",
        "source_units": ("mmhg", "unitless"),
        "canonical_item": "标准状态下氧饱和度50%时的氧分压",
        "canonical_unit": "mmhg",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Canonical standard-condition P50.",
    },
    {
        "rule_id": "patient_condition_p50_identity",
        "source_item": "患者体温下氧饱和度50%时的氧分压",
        "source_units": ("mmhg", "unitless"),
        "canonical_item": "患者体温下氧饱和度50%时的氧分压",
        "canonical_unit": "mmhg",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Canonical patient-condition P50.",
    },
    {
        "rule_id": "patient_condition_p50_generic_alias",
        "source_item": "氧饱和度50%时的氧分压",
        "source_units": ("mmhg", "unitless"),
        "canonical_item": "患者体温下氧饱和度50%时的氧分压",
        "canonical_unit": "mmhg",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Same-time values equal patient-condition P50 in 99.98% of pairs.",
    },
    {
        "rule_id": "patient_condition_p50_oxygen_concentration_alias",
        "source_item": "血氧浓度50%氧分压",
        "source_units": ("mmhg", "unitless"),
        "canonical_item": "患者体温下氧饱和度50%时的氧分压",
        "canonical_unit": "mmhg",
        "factor": 1.0,
        "formula": "identity",
        "rationale": "Same-time values equal patient-condition P50 in 97.84% of pairs.",
    },
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_unit(value):
    text = str(value).strip()
    if not text:
        return "unitless"
    return re.sub(r"\s+", "", text).lower()


def _extract_numeric(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(
        cleaned.str.extract(
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        )[0],
        errors="coerce",
    )


def _join_unique(values):
    return "^".join(sorted({str(value) for value in values if str(value)}))


def _harmonize_analytes(data):
    data = data.copy()
    data["source_item_name"] = data["item_name"]
    data["source_unit"] = data["unit"]
    data["source_numeric_value"] = data["numeric_value"]
    data["harmonization_rule"] = "identity_unmodified"
    audit_rows = []
    for rule in HARMONIZATION_RULES:
        mask = (
            data["source_item_name"].eq(rule["source_item"])
            & data["source_unit"].isin(rule["source_units"])
        )
        selected = data.loc[mask]
        if selected.empty:
            continue
        converted = selected["source_numeric_value"] * rule["factor"]
        data.loc[mask, "numeric_value"] = converted
        data.loc[mask, "item_name"] = rule["canonical_item"]
        data.loc[mask, "unit"] = rule["canonical_unit"]
        data.loc[mask, "harmonization_rule"] = rule["rule_id"]
        audit_rows.append(
            {
                "rule_id": rule["rule_id"],
                "source_item_name": rule["source_item"],
                "source_units": "^".join(rule["source_units"]),
                "canonical_item_name": rule["canonical_item"],
                "canonical_unit": rule["canonical_unit"],
                "value_formula": rule["formula"],
                "conversion_factor": rule["factor"],
                "rows_affected": int(len(selected)),
                "patients_affected": int(selected["hospital_id"].nunique()),
                "source_value_min": float(selected["source_numeric_value"].min()),
                "source_value_median": float(
                    selected["source_numeric_value"].median()
                ),
                "source_value_max": float(selected["source_numeric_value"].max()),
                "canonical_value_min": float(converted.min()),
                "canonical_value_median": float(converted.median()),
                "canonical_value_max": float(converted.max()),
                "rationale": rule["rationale"],
            }
        )
    return data, pd.DataFrame(audit_rows)


def _field_equivalence_evidence(data):
    index_columns = [
        "hospital_id",
        "admission_time",
        "discharge_time",
        "report_time",
    ]
    source = data.copy()
    source["source_key"] = (
        source["source_item_name"] + " [" + source["source_unit"] + "]"
    )
    wide = source.pivot_table(
        index=index_columns,
        columns="source_key",
        values="source_numeric_value",
        aggfunc="median",
    )
    rows = []

    def equality_check(check_id, left, right, conclusion, action):
        paired = wide[[left, right]].dropna()
        difference = paired[left] - paired[right]
        rows.append(
            {
                "check_id": check_id,
                "evidence_type": "same_timestamp_equality",
                "left_field": left,
                "right_field_or_formula": right,
                "paired_rows": int(len(paired)),
                "patients": int(
                    paired.reset_index()["hospital_id"].nunique()
                ),
                "spearman_rho": float(
                    paired[left].corr(paired[right], method="spearman")
                ),
                "median_abs_error": float(np.median(np.abs(difference))),
                "median_relative_error": float(
                    np.median(
                        np.abs(difference)
                        / np.maximum(
                            (np.abs(paired[left]) + np.abs(paired[right])) / 2,
                            1e-12,
                        )
                    )
                ),
                "within_tolerance_pct": float(
                    np.isclose(
                        paired[left],
                        paired[right],
                        rtol=1e-6,
                        atol=1e-6,
                    ).mean()
                    * 100
                ),
                "conclusion": conclusion,
                "action": action,
            }
        )

    def formula_check(
        check_id,
        observed,
        numerator,
        denominator,
        factor,
        conclusion,
        action,
    ):
        paired = wide[[observed, numerator, denominator]].dropna()
        expected = paired[numerator] / paired[denominator] * factor
        difference = paired[observed] - expected
        rows.append(
            {
                "check_id": check_id,
                "evidence_type": "same_timestamp_formula",
                "left_field": observed,
                "right_field_or_formula": (
                    f"{factor:g} * {numerator} / {denominator}"
                ),
                "paired_rows": int(len(paired)),
                "patients": int(
                    paired.reset_index()["hospital_id"].nunique()
                ),
                "spearman_rho": float(
                    paired[observed].corr(expected, method="spearman")
                ),
                "median_abs_error": float(np.median(np.abs(difference))),
                "median_relative_error": float(
                    np.median(
                        np.abs(difference)
                        / np.maximum(
                            (np.abs(paired[observed]) + np.abs(expected)) / 2,
                            1e-12,
                        )
                    )
                ),
                "within_tolerance_pct": float(
                    np.isclose(
                        paired[observed],
                        expected,
                        rtol=0.01,
                        atol=0.05,
                    ).mean()
                    * 100
                ),
                "conclusion": conclusion,
                "action": action,
            }
        )

    formula_check(
        "arterial_alveolar_percent_formula",
        "动脉氧分压与肺泡氧分压之比 [%]",
        "氧分压 [mmhg]",
        "平均肺泡氧分压 [mmhg]",
        100.0,
        "The percentage field is 100 * PO2 / alveolar PO2.",
        "Retain as canonical percent representation.",
    )
    formula_check(
        "arterial_alveolar_fraction_formula",
        "动脉血氧分压与肺泡内氧分压之比 [unitless]",
        "氧分压 [mmhg]",
        "肺泡内氧分压 [mmhg]",
        1.0,
        "The unitless field is PO2 / alveolar PO2.",
        "Multiply by 100 and merge with the percentage field.",
    )
    equality_check(
        "patient_p50_generic_alias",
        "患者体温下氧饱和度50%时的氧分压 [mmhg]",
        "氧饱和度50%时的氧分压 [mmhg]",
        "Fields are effectively identical at the same report time.",
        "Deduplicate as patient-condition P50.",
    )
    equality_check(
        "patient_p50_oxygen_concentration_alias",
        "患者体温下氧饱和度50%时的氧分压 [mmhg]",
        "血氧浓度50%氧分压 [mmhg]",
        "Fields are effectively identical at the same report time.",
        "Deduplicate as patient-condition P50.",
    )

    mmol = source[
        source["source_key"].eq("葡萄糖浓度 [mmol/l]")
    ][index_columns + ["source_numeric_value"]].rename(
        columns={"source_numeric_value": "mmol_l"}
    )
    mg = source[
        source["source_key"].eq("葡萄糖浓度 [mg/dl]")
    ][index_columns + ["source_numeric_value"]].rename(
        columns={"source_numeric_value": "mg_dl"}
    )
    mmol = mmol.sort_values("report_time")
    mg = mg.sort_values("report_time")
    nearest = pd.merge_asof(
        mmol,
        mg,
        on="report_time",
        by=["hospital_id", "admission_time", "discharge_time"],
        direction="nearest",
        tolerance=pd.Timedelta(hours=24),
    ).dropna(subset=["mg_dl"])
    converted = nearest["mmol_l"] * MG_DL_PER_MMOL_L_GLUCOSE
    relative_error = np.abs(converted - nearest["mg_dl"]) / np.maximum(
        (np.abs(converted) + np.abs(nearest["mg_dl"])) / 2,
        1e-12,
    )
    rows.append(
        {
            "check_id": "blood_gas_glucose_unit_conversion",
            "evidence_type": "nearest_within_24h_unit_conversion",
            "left_field": "葡萄糖浓度 [mmol/l] * 18.0182",
            "right_field_or_formula": "葡萄糖浓度 [mg/dl]",
            "paired_rows": int(len(nearest)),
            "patients": int(nearest["hospital_id"].nunique()),
            "spearman_rho": float(
                converted.corr(nearest["mg_dl"], method="spearman")
            ),
            "median_abs_error": float(
                np.median(np.abs(converted - nearest["mg_dl"]))
            ),
            "median_relative_error": float(np.median(relative_error)),
            "within_tolerance_pct": float(
                np.mean(relative_error <= 0.20) * 100
            ),
            "conclusion": (
                "Ranges and temporally adjacent values are compatible with "
                "the standard glucose unit conversion; panels do not report "
                "both units at the exact same timestamp."
            ),
            "action": "Convert mg/dL to mmol/L and merge.",
        }
    )
    rows.append(
        {
            "check_id": "standard_condition_p50_semantic_alias",
            "evidence_type": "definition_and_range",
            "left_field": (
                "在PH7.4,PCO2=40mmHg,体温37度SO2=50%的氧分压 [mmhg]"
            ),
            "right_field_or_formula": (
                "标准状态下氧饱和度50%时的氧分压 [mmhg]"
            ),
            "paired_rows": 0,
            "patients": int(
                source.loc[
                    source["source_item_name"].isin(
                        (
                            "在PH7.4,PCO2=40mmHg,体温37度SO2=50%的氧分压",
                            "标准状态下氧饱和度50%时的氧分压",
                        )
                    ),
                    "hospital_id",
                ].nunique()
            ),
            "spearman_rho": np.nan,
            "median_abs_error": np.nan,
            "median_relative_error": np.nan,
            "within_tolerance_pct": np.nan,
            "conclusion": (
                "Both labels define P50 at pH 7.4, PCO2 40 mmHg and 37 C; "
                "their source panels are mutually exclusive."
            ),
            "action": (
                "Merge as standard-condition P50; keep separate from "
                "patient-condition P50."
            ),
        }
    )
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    order = valid[np.argsort(values[valid], kind="stable")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result


def _rank_biserial(changes):
    changes = np.asarray(changes, dtype=np.float64)
    changes = changes[np.isfinite(changes) & ~np.isclose(changes, 0.0)]
    if not len(changes):
        return 0.0
    ranks = rankdata(np.abs(changes), method="average")
    positive = ranks[changes > 0].sum()
    negative = ranks[changes < 0].sum()
    return float((positive - negative) / (positive + negative))


def _bootstrap_median_ci(values, rng):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        estimates[index] = np.median(
            rng.choice(values, size=len(values), replace=True)
        )
    return tuple(np.quantile(estimates, (0.025, 0.975)).astype(float))


def _safe_wilcoxon(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.allclose(values, 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def _load_and_clean(source_path):
    raw = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
        usecols=list(REQUIRED_COLUMNS),
    )
    flow = [("Raw rows", len(raw))]
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["admission_time"] = pd.to_datetime(
        raw["首页入院时间"], errors="coerce"
    )
    raw["discharge_time"] = pd.to_datetime(
        raw["首页出院时间"], errors="coerce"
    )
    raw["report_time"] = pd.to_datetime(raw["报告时间"], errors="coerce")
    valid_episode = (
        raw["hospital_id"].ne("")
        & raw["admission_time"].notna()
        & raw["discharge_time"].notna()
        & raw["report_time"].notna()
        & raw["discharge_time"].gt(raw["admission_time"])
    )
    flow.append(("Valid ID and episode times", int(valid_episode.sum())))
    in_stay = (
        valid_episode
        & raw["report_time"].ge(raw["admission_time"])
        & raw["report_time"].le(raw["discharge_time"])
    )
    flow.append(("Reports during hospitalization", int(in_stay.sum())))
    raw["numeric_value"] = _extract_numeric(raw["检验值(文本)"])
    numeric = in_stay & raw["numeric_value"].notna()
    flow.append(("Numeric in-stay results", int(numeric.sum())))
    raw["is_censored"] = raw["检验值(文本)"].str.match(
        r"^\s*[<>≤≥＜＞]", na=False
    )
    uncensored = numeric & ~raw["is_censored"]
    flow.append(("Uncensored numeric results", int(uncensored.sum())))

    data = raw.loc[uncensored].copy()
    data["item_name"] = data["检验项名称"].astype(str).str.strip()
    data["unit"] = data["单位"].map(_normalize_unit)
    data = data[data["item_name"].ne("")].copy()
    flow.append(("Named analyte-unit results", int(len(data))))
    data, harmonization_audit = _harmonize_analytes(data)
    equivalence_evidence = _field_equivalence_evidence(data)
    flow.append(("Harmonized analyte results", int(len(data))))
    data["episode_key"] = (
        data["hospital_id"]
        + "|"
        + data["admission_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "|"
        + data["discharge_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    episode_keys = sorted(data["episode_key"].unique())
    episode_lookup = {
        key: f"E{index + 1:04d}" for index, key in enumerate(episode_keys)
    }
    data["episode_id"] = data["episode_key"].map(episode_lookup)
    variable_keys = sorted(
        set(zip(data["item_name"].astype(str), data["unit"].astype(str)))
    )
    variable_lookup = {
        key: f"A{index + 1:03d}" for index, key in enumerate(variable_keys)
    }
    data["variable_id"] = [
        variable_lookup[(item, unit)]
        for item, unit in zip(data["item_name"], data["unit"])
    ]
    data["stay_duration_days"] = (
        data["discharge_time"] - data["admission_time"]
    ).dt.total_seconds() / 86400.0
    data["elapsed_days"] = (
        data["report_time"] - data["admission_time"]
    ).dt.total_seconds() / 86400.0
    data["stay_progress"] = (
        data["elapsed_days"] / data["stay_duration_days"]
    ).clip(0.0, 1.0)
    data["progress_bin"] = np.minimum(
        (data["stay_progress"] * PROGRESS_BIN_COUNT).astype(int),
        PROGRESS_BIN_COUNT - 1,
    )
    data["hospital_day_bin"] = np.minimum(
        np.floor(data["elapsed_days"]).astype(int), 15
    )

    collapsed = (
        data.groupby(
            [
                "hospital_id",
                "episode_id",
                "admission_time",
                "discharge_time",
                "stay_duration_days",
                "variable_id",
                "item_name",
                "unit",
                "report_time",
            ],
            as_index=False,
        )
        .agg(
            numeric_value=("numeric_value", "median"),
            duplicate_rows=("numeric_value", "size"),
            source_item_names=("source_item_name", _join_unique),
            source_units=("source_unit", _join_unique),
            harmonization_rules=("harmonization_rule", _join_unique),
            elapsed_days=("elapsed_days", "first"),
            stay_progress=("stay_progress", "first"),
            progress_bin=("progress_bin", "first"),
            hospital_day_bin=("hospital_day_bin", "first"),
        )
        .sort_values(["episode_id", "variable_id", "report_time"])
        .reset_index(drop=True)
    )
    flow.append(("Collapsed episode-time values", int(len(collapsed))))

    dictionary_rows = []
    source_summary = (
        data.groupby(["item_name", "unit"])
        .agg(
            source_item_names=("source_item_name", _join_unique),
            source_units=("source_unit", _join_unique),
            harmonization_rules=("harmonization_rule", _join_unique),
        )
        .reset_index()
        .set_index(["item_name", "unit"])
    )
    for (item, unit), variable_id in variable_lookup.items():
        english = ENGLISH_NAMES.get(item, item)
        source = source_summary.loc[(item, unit)]
        dictionary_rows.append(
            {
                "variable_id": variable_id,
                "item_name_cn": item,
                "item_name_en": english,
                "unit": unit,
                "plot_label": f"{variable_id} {english} [{unit}]",
                "source_item_names": source["source_item_names"],
                "source_units": source["source_units"],
                "harmonization_rules": source["harmonization_rules"],
            }
        )
    dictionary = pd.DataFrame(dictionary_rows).sort_values("variable_id")
    exclusions = {
        "invalid_id_or_episode_time_rows": int((~valid_episode).sum()),
        "reports_before_admission": int(
            (valid_episode & raw["report_time"].lt(raw["admission_time"])).sum()
        ),
        "reports_after_discharge": int(
            (valid_episode & raw["report_time"].gt(raw["discharge_time"])).sum()
        ),
        "nonnumeric_in_stay_rows": int((in_stay & raw["numeric_value"].isna()).sum()),
        "censored_numeric_in_stay_rows": int(
            (numeric & raw["is_censored"]).sum()
        ),
        "same_timestamp_duplicate_rows_collapsed": int(
            len(data) - len(collapsed)
        ),
        "same_timestamp_alias_groups_collapsed": int(
            (
                collapsed["source_item_names"].str.contains(
                    r"\^", regex=True, na=False
                )
                | collapsed["source_units"].str.contains(
                    r"\^", regex=True, na=False
                )
            ).sum()
        ),
    }
    return (
        raw,
        collapsed,
        dictionary,
        pd.DataFrame(flow, columns=("stage", "rows")),
        exclusions,
        harmonization_audit,
        equivalence_evidence,
    )


def _coverage(measurements, dictionary):
    group_counts = (
        measurements.groupby(["variable_id", "episode_id"])
        .size()
        .rename("episode_measurements")
        .reset_index()
    )
    coverage = (
        measurements.groupby("variable_id")
        .agg(
            measurements=("numeric_value", "size"),
            patients=("hospital_id", "nunique"),
            episodes=("episode_id", "nunique"),
            value_min=("numeric_value", "min"),
            value_median=("numeric_value", "median"),
            value_max=("numeric_value", "max"),
        )
        .reset_index()
    )
    repeated = (
        group_counts[group_counts["episode_measurements"].ge(2)]
        .groupby("variable_id")
        .agg(
            repeated_episodes=("episode_id", "nunique"),
            median_measurements_per_repeated_episode=(
                "episode_measurements",
                "median",
            ),
        )
        .reset_index()
    )
    coverage = coverage.merge(repeated, on="variable_id", how="left")
    coverage[["repeated_episodes", "median_measurements_per_repeated_episode"]] = (
        coverage[
            ["repeated_episodes", "median_measurements_per_repeated_episode"]
        ].fillna(0)
    )
    coverage["eligible_longitudinal"] = coverage["repeated_episodes"].ge(
        MIN_REPEATED_EPISODES
    )
    return coverage.merge(dictionary, on="variable_id", how="left")


def _episode_changes(measurements, eligible_ids):
    records = []
    slopes = []
    selected = measurements[measurements["variable_id"].isin(eligible_ids)]
    for (variable_id, episode_id), group in selected.groupby(
        ["variable_id", "episode_id"], sort=True
    ):
        group = group.sort_values("report_time")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        change = float(last["numeric_value"] - first["numeric_value"])
        relative = (
            change / abs(float(first["numeric_value"])) * 100.0
            if not np.isclose(float(first["numeric_value"]), 0.0)
            else np.nan
        )
        records.append(
            {
                "hospital_id": str(first["hospital_id"]),
                "episode_id": episode_id,
                "variable_id": variable_id,
                "measurement_count": int(len(group)),
                "first_time": first["report_time"],
                "last_time": last["report_time"],
                "followup_days": float(
                    (last["report_time"] - first["report_time"]).total_seconds()
                    / 86400.0
                ),
                "first_value": float(first["numeric_value"]),
                "last_value": float(last["numeric_value"]),
                "absolute_change": change,
                "relative_change_pct": relative,
            }
        )
        unique_times = group.groupby("elapsed_days", as_index=False)[
            "numeric_value"
        ].median()
        if len(unique_times) >= 3 and unique_times["elapsed_days"].nunique() >= 3:
            slope = float(
                theilslopes(
                    unique_times["numeric_value"].to_numpy(np.float64),
                    unique_times["elapsed_days"].to_numpy(np.float64),
                ).slope
            )
            slopes.append(
                {
                    "hospital_id": str(first["hospital_id"]),
                    "episode_id": episode_id,
                    "variable_id": variable_id,
                    "measurement_count": int(len(group)),
                    "theil_sen_slope_per_day": slope,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(slopes)


def _change_statistics(changes, measurements, dictionary):
    patient_changes = (
        changes.groupby(["variable_id", "hospital_id"], as_index=False)
        .agg(
            patient_median_change=("absolute_change", "median"),
            patient_median_relative_change_pct=("relative_change_pct", "median"),
            episode_count=("episode_id", "nunique"),
        )
    )
    rng = np.random.default_rng(SEED)
    rows = []
    for variable_id, group in patient_changes.groupby("variable_id", sort=True):
        if len(group) < MIN_PAIRED_PATIENTS:
            continue
        values = group["patient_median_change"].to_numpy(np.float64)
        measurement_values = measurements.loc[
            measurements["variable_id"].eq(variable_id), "numeric_value"
        ].to_numpy(np.float64)
        iqr = float(
            np.quantile(measurement_values, 0.75)
            - np.quantile(measurement_values, 0.25)
        )
        scale_iqr = iqr if iqr > 0 else np.nan
        statistic, p_value = _safe_wilcoxon(values)
        ci_low, ci_high = _bootstrap_median_ci(values, rng)
        rows.append(
            {
                "variable_id": variable_id,
                "paired_episodes": int(
                    changes.loc[
                        changes["variable_id"].eq(variable_id), "episode_id"
                    ].nunique()
                ),
                "paired_patients": int(len(group)),
                "median_first_last_change": float(np.median(values)),
                "median_change_ci95_low": ci_low,
                "median_change_ci95_high": ci_high,
                "median_relative_change_pct": float(
                    np.nanmedian(
                        group["patient_median_relative_change_pct"].to_numpy(
                            np.float64
                        )
                    )
                ),
                "measurement_iqr": iqr,
                "standardized_median_change_iqr": float(
                    np.median(values) / scale_iqr
                ),
                "standardized_ci95_low": ci_low / scale_iqr,
                "standardized_ci95_high": ci_high / scale_iqr,
                "rank_biserial": _rank_biserial(values),
                "increase_patients": int(np.count_nonzero(values > 0)),
                "decrease_patients": int(np.count_nonzero(values < 0)),
                "unchanged_patients": int(np.count_nonzero(np.isclose(values, 0))),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["q_value_bh"] = _benjamini_hochberg(statistics["p_value"])
    statistics["fdr_significant_0_05"] = statistics["q_value_bh"].le(0.05)
    statistics = statistics.merge(dictionary, on="variable_id", how="left")
    return patient_changes, statistics


def _slope_statistics(slopes, measurements, dictionary):
    if slopes.empty:
        return pd.DataFrame()
    patient_slopes = (
        slopes.groupby(["variable_id", "hospital_id"], as_index=False)[
            "theil_sen_slope_per_day"
        ]
        .median()
        .rename(columns={"theil_sen_slope_per_day": "patient_median_slope_per_day"})
    )
    rows = []
    for variable_id, group in patient_slopes.groupby("variable_id", sort=True):
        values = group["patient_median_slope_per_day"].to_numpy(np.float64)
        measurement_values = measurements.loc[
            measurements["variable_id"].eq(variable_id), "numeric_value"
        ].to_numpy(np.float64)
        iqr = float(
            np.quantile(measurement_values, 0.75)
            - np.quantile(measurement_values, 0.25)
        )
        scale_iqr = iqr if iqr > 0 else np.nan
        statistic, p_value = _safe_wilcoxon(values)
        rows.append(
            {
                "variable_id": variable_id,
                "slope_episodes": int(
                    slopes.loc[
                        slopes["variable_id"].eq(variable_id), "episode_id"
                    ].nunique()
                ),
                "slope_patients": int(len(group)),
                "median_slope_per_day": float(np.median(values)),
                "standardized_median_slope_iqr_per_day": float(
                    np.median(values) / scale_iqr
                ),
                "rank_biserial": _rank_biserial(values),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )
    statistics = pd.DataFrame(rows)
    statistics["q_value_bh"] = _benjamini_hochberg(statistics["p_value"])
    statistics["fdr_significant_0_05"] = statistics["q_value_bh"].le(0.05)
    return statistics.merge(dictionary, on="variable_id", how="left")


def _binned_trajectories(measurements, eligible_ids, dictionary):
    selected = measurements[measurements["variable_id"].isin(eligible_ids)].copy()
    patient_bins = (
        selected.groupby(
            ["variable_id", "hospital_id", "progress_bin"], as_index=False
        )
        .agg(
            patient_bin_value=("numeric_value", "median"),
            episode_count=("episode_id", "nunique"),
        )
    )
    summary = (
        patient_bins.groupby(["variable_id", "progress_bin"])
        .agg(
            patients=("hospital_id", "nunique"),
            median=("patient_bin_value", "median"),
            q25=("patient_bin_value", lambda x: np.quantile(x, 0.25)),
            q75=("patient_bin_value", lambda x: np.quantile(x, 0.75)),
        )
        .reset_index()
    )
    summary["progress_midpoint"] = (
        summary["progress_bin"] + 0.5
    ) / PROGRESS_BIN_COUNT
    scale = (
        selected.groupby("variable_id")["numeric_value"]
        .agg(
            global_median="median",
            global_q25=lambda x: np.quantile(x, 0.25),
            global_q75=lambda x: np.quantile(x, 0.75),
        )
        .reset_index()
    )
    scale["global_iqr"] = scale["global_q75"] - scale["global_q25"]
    scale.loc[scale["global_iqr"].le(0), "global_iqr"] = np.nan
    summary = summary.merge(scale, on="variable_id", how="left")
    summary["robust_z_median"] = (
        summary["median"] - summary["global_median"]
    ) / summary["global_iqr"]
    baseline = (
        summary.sort_values("progress_bin")
        .groupby("variable_id", as_index=False)["robust_z_median"]
        .first()
        .rename(columns={"robust_z_median": "first_bin_robust_z"})
    )
    summary = summary.merge(baseline, on="variable_id", how="left")
    summary["change_from_first_bin_iqr"] = (
        summary["robust_z_median"] - summary["first_bin_robust_z"]
    )
    return summary.merge(dictionary, on="variable_id", how="left")


def _figure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(figure, path):
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _output_directories(output_dir):
    directories = {
        "figures": os.path.join(output_dir, "figures"),
        "tables": os.path.join(output_dir, "tables"),
        "reports": os.path.join(output_dir, "reports"),
        "metadata": os.path.join(output_dir, "metadata"),
    }
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
    return directories


def _clean_legacy_output_layout(output_dir):
    legacy_extensions = {".csv", ".json", ".md", ".pdf", ".png"}
    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path) and os.path.splitext(filename)[1] in legacy_extensions:
            os.remove(path)
    legacy_per_analyte = os.path.join(output_dir, "per_analyte")
    if os.path.isdir(legacy_per_analyte):
        shutil.rmtree(legacy_per_analyte)


def _plot_cohort_flow(flow, output_dir):
    figure, axis = plt.subplots(figsize=(10, 5.5))
    values = flow["rows"].to_numpy()
    labels = flow["stage"].tolist()
    bars = axis.barh(range(len(flow)), values, color="#2F6B8A")
    axis.set_yticks(range(len(flow)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Rows")
    axis.set_title("Laboratory data inclusion flow")
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_width() + values.max() * 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value):,}",
            va="center",
        )
    _save_figure(figure, os.path.join(output_dir, "cohort_flow.png"))


def _plot_coverage(coverage, output_dir):
    shown = coverage.sort_values("repeated_episodes", ascending=True)
    figure, axis = plt.subplots(figsize=(12, max(7, 0.34 * len(shown))))
    y = np.arange(len(shown))
    axis.barh(
        y,
        shown["episodes"],
        color="#B8C4CC",
        label="Any measurement",
    )
    axis.barh(
        y,
        shown["repeated_episodes"],
        color="#2F6B8A",
        label="At least 2 measurements",
    )
    axis.set_yticks(y, shown["plot_label"], fontsize=7)
    axis.set_xlabel("Hospital episodes")
    axis.set_title("Analyte coverage by hospital episode")
    axis.legend(loc="lower right")
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, os.path.join(output_dir, "analyte_coverage.png"))


def _plot_timing(measurements, output_dir):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(
        measurements["stay_progress"],
        bins=np.linspace(0, 1, 21),
        color="#2F6B8A",
        edgecolor="white",
    )
    axes[0].set_xlabel("Fraction of hospital stay")
    axes[0].set_ylabel("Collapsed measurements")
    axes[0].set_title("Measurement timing over normalized stay")
    day_counts = (
        measurements["hospital_day_bin"].value_counts().sort_index()
    )
    labels = [str(x) if x < 15 else "15+" for x in day_counts.index]
    axes[1].bar(labels, day_counts.values, color="#C46A45")
    axes[1].set_xlabel("Hospital day")
    axes[1].set_ylabel("Collapsed measurements")
    axes[1].set_title("Measurement timing by hospital day")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, os.path.join(output_dir, "measurement_timing.png"))


def _plot_change_forest(statistics, output_dir):
    shown = statistics[
        np.isfinite(statistics["standardized_median_change_iqr"])
        & np.isfinite(statistics["standardized_ci95_low"])
        & np.isfinite(statistics["standardized_ci95_high"])
    ].sort_values("standardized_median_change_iqr")
    figure, axis = plt.subplots(figsize=(12, max(7, 0.38 * len(shown))))
    y = np.arange(len(shown))
    values = shown["standardized_median_change_iqr"].to_numpy()
    lower = values - shown["standardized_ci95_low"].to_numpy()
    upper = shown["standardized_ci95_high"].to_numpy() - values
    colors = np.where(
        shown["fdr_significant_0_05"], "#C44E52", "#6C7A86"
    )
    for index in range(len(shown)):
        axis.errorbar(
            values[index],
            y[index],
            xerr=np.array([[lower[index]], [upper[index]]]),
            fmt="o",
            color=colors[index],
            capsize=2,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(y, shown["plot_label"], fontsize=7)
    axis.set_xlabel("Median first-to-last change / overall IQR")
    axis.set_title("Patient-level first-to-last change with bootstrap 95% CI")
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, os.path.join(output_dir, "paired_change_forest.png"))


def _plot_significance(statistics, output_dir):
    statistics = statistics[
        np.isfinite(statistics["standardized_median_change_iqr"])
    ].copy()
    figure, axis = plt.subplots(figsize=(9, 6))
    q_values = statistics["q_value_bh"].clip(lower=1e-300)
    colors = np.where(
        statistics["fdr_significant_0_05"], "#C44E52", "#6C7A86"
    )
    axis.scatter(
        statistics["standardized_median_change_iqr"],
        -np.log10(q_values),
        c=colors,
        s=42,
        alpha=0.9,
    )
    for row in statistics.itertuples(index=False):
        if row.fdr_significant_0_05 or abs(row.standardized_median_change_iqr) >= 0.5:
            axis.annotate(
                row.variable_id,
                (row.standardized_median_change_iqr, -math.log10(max(row.q_value_bh, 1e-300))),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Median first-to-last change / overall IQR")
    axis.set_ylabel("-log10(BH-FDR q)")
    axis.set_title("Magnitude and evidence for inpatient change")
    axis.grid(alpha=0.2)
    _save_figure(figure, os.path.join(output_dir, "change_significance.png"))


def _plot_heatmap(trajectories, statistics, output_dir):
    order = statistics[
        np.isfinite(statistics["standardized_median_change_iqr"])
    ].sort_values(
        "standardized_median_change_iqr", ascending=False
    )["variable_id"].tolist()
    matrix = trajectories.pivot(
        index="variable_id",
        columns="progress_bin",
        values="change_from_first_bin_iqr",
    ).reindex(index=order, columns=range(PROGRESS_BIN_COUNT))
    labels = (
        statistics.set_index("variable_id").loc[order, "plot_label"].tolist()
    )
    values = matrix.to_numpy(np.float64)
    finite = np.abs(values[np.isfinite(values)])
    limit = max(float(np.quantile(finite, 0.95)) if len(finite) else 1.0, 0.25)
    figure, axis = plt.subplots(figsize=(12, max(7, 0.36 * len(order))))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_xticks(
        np.arange(PROGRESS_BIN_COUNT),
        [f"{10 * i}-{10 * (i + 1)}%" for i in range(PROGRESS_BIN_COUNT)],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Normalized hospital-stay interval")
    axis.set_title("Patient-balanced trajectory relative to first observed stay bin")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Median change / analyte IQR")
    figure.tight_layout()
    _save_figure(figure, os.path.join(output_dir, "longitudinal_heatmap.png"))


def _plot_top_trajectories(trajectories, statistics, output_dir):
    ranked = statistics.assign(
        magnitude=statistics["standardized_median_change_iqr"].abs()
    ).sort_values("magnitude", ascending=False).head(12)
    figure, axes = plt.subplots(4, 3, figsize=(15, 14), squeeze=False)
    for axis, row in zip(axes.flat, ranked.itertuples(index=False)):
        values = trajectories[
            trajectories["variable_id"].eq(row.variable_id)
        ].sort_values("progress_bin")
        x = values["progress_midpoint"] * 100
        axis.plot(x, values["median"], color="#2F6B8A", marker="o")
        axis.fill_between(
            x,
            values["q25"],
            values["q75"],
            color="#8CB7C9",
            alpha=0.35,
        )
        for point_x, point_y, patients in zip(
            x,
            values["median"],
            values["patients"],
        ):
            axis.annotate(
                f"n={int(patients)}",
                (point_x, point_y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#244F65",
            )
        axis.set_title(f"{row.variable_id} {row.item_name_en}", fontsize=9)
        axis.set_xlabel("Hospital stay (%)")
        axis.set_ylabel(row.unit)
        axis.margins(y=0.12)
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Largest patient-level first-to-last changes: median and IQR trajectories",
        fontsize=14,
    )
    figure.tight_layout()
    _save_figure(figure, os.path.join(output_dir, "top_trajectory_panels.png"))


def _plot_per_analyte(
    measurements,
    changes,
    slopes,
    trajectories,
    statistics,
    output_dir,
):
    per_dir = os.path.join(output_dir, "per_analyte")
    os.makedirs(per_dir, exist_ok=True)
    expected_pngs = {
        f"{variable_id}.png"
        for variable_id in statistics["variable_id"].astype(str)
    }
    for filename in os.listdir(per_dir):
        if filename.endswith(".png") and filename not in expected_pngs:
            os.remove(os.path.join(per_dir, filename))
    paths = []
    pdf_path = os.path.join(output_dir, "all_analyte_trajectories.pdf")
    with PdfPages(pdf_path) as pdf:
        for row in statistics.sort_values("variable_id").itertuples(index=False):
            variable_id = row.variable_id
            trajectory = trajectories[
                trajectories["variable_id"].eq(variable_id)
            ].sort_values("progress_bin")
            episode_change = changes[
                changes["variable_id"].eq(variable_id)
            ]
            slope_values = slopes.loc[
                slopes["variable_id"].eq(variable_id),
                "theil_sen_slope_per_day",
            ].to_numpy(np.float64)
            figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
            x = trajectory["progress_midpoint"] * 100
            axes[0].plot(x, trajectory["median"], color="#2F6B8A", marker="o")
            axes[0].fill_between(
                x,
                trajectory["q25"],
                trajectory["q75"],
                color="#8CB7C9",
                alpha=0.35,
            )
            axes[0].set_xlabel("Hospital stay (%)")
            axes[0].set_ylabel(row.unit)
            axes[0].set_title("Patient-balanced median and IQR")

            paired = episode_change[["first_value", "last_value"]].to_numpy(
                np.float64
            )
            if len(paired):
                rng = np.random.default_rng(SEED + int(variable_id[1:]))
                selected = (
                    rng.choice(len(paired), size=min(80, len(paired)), replace=False)
                    if len(paired) > 80
                    else np.arange(len(paired))
                )
                for values in paired[selected]:
                    axes[1].plot((0, 1), values, color="#97A4AD", alpha=0.18)
                axes[1].boxplot(
                    [paired[:, 0], paired[:, 1]],
                    positions=(0, 1),
                    tick_labels=("First", "Last"),
                    showfliers=False,
                )
            axes[1].set_ylabel(row.unit)
            shown_pairs = min(80, len(episode_change))
            axes[1].set_title(
                "Episode first/last values "
                f"(boxplots n={len(episode_change)}, lines n={shown_pairs})"
            )

            if len(slope_values):
                axes[2].hist(
                    slope_values,
                    bins=25,
                    color="#C46A45",
                    edgecolor="white",
                )
                axes[2].axvline(0, color="black", linewidth=1)
                axes[2].axvline(
                    np.median(slope_values),
                    color="#2F6B8A",
                    linestyle="--",
                    linewidth=1.5,
                )
            axes[2].set_xlabel(f"{row.unit} per day")
            axes[2].set_title(f"Theil-Sen episode slopes (n={len(slope_values)})")
            for axis in axes:
                axis.grid(alpha=0.2)
            figure.suptitle(
                f"{row.variable_id} {row.item_name_en} [{row.unit}]\n"
                f"Median patient change={row.median_first_last_change:.4g}; "
                f"BH q={row.q_value_bh:.3g}",
                fontsize=13,
            )
            figure.tight_layout()
            path = os.path.join(per_dir, f"{variable_id}.png")
            figure.savefig(path, dpi=170, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            paths.append(
                {
                    "variable_id": variable_id,
                    "png_path": os.path.relpath(path, output_dir),
                    "pdf_path": os.path.basename(pdf_path),
                }
            )
    return pd.DataFrame(paths)


def _write_report(
    output_dir,
    source_path,
    measurements,
    dictionary,
    coverage,
    changes,
    statistics,
    slope_statistics,
    exclusions,
    surgery_summary,
    harmonization_audit,
    equivalence_evidence,
):
    significant = statistics[statistics["fdr_significant_0_05"]].copy()
    increases = significant.sort_values(
        "standardized_median_change_iqr", ascending=False
    ).head(8)
    decreases = significant.sort_values(
        "standardized_median_change_iqr", ascending=True
    ).head(8)

    def rows(frame):
        output = []
        for row in frame.itertuples(index=False):
            output.append(
                f"| {row.variable_id} | {row.item_name_cn} | {row.unit} | "
                f"{row.median_first_last_change:.4g} | "
                f"{row.standardized_median_change_iqr:.3f} | "
                f"{row.q_value_bh:.3g} |"
            )
        return "\n".join(output) if output else "| - | - | - | - | - | - |"

    text = f"""# 住院期化验值纵向变化统计报告

## 数据与口径

- 数据源：`{source_path}`
- 清洗后患者：{measurements['hospital_id'].nunique()} 人
- 清洗后住院 episode：{measurements['episode_id'].nunique()} 次
- 同时间重复值合并后数值测量：{len(measurements):,} 条
- 检验项-单位组合：{len(dictionary)} 个
- 满足至少 {MIN_REPEATED_EPISODES} 次重复测量住院的纵向变量：{int(coverage['eligible_longitudinal'].sum())} 个
- 进入首末配对推断的变量：{len(statistics)} 个
- BH-FDR q<0.05 的变量：{len(significant)} 个

一次住院由病案号、入院时间和出院时间共同定义。仅保留报告时间位于住院区间内的非截尾数值结果；同一住院、同一规范化变量、同一报告时间的重复或别名行取中位数。除下述经过验证的换算与别名规则外，检验项仍按名称和单位分开。

## 等价字段规范化

本次分析在生成变量 ID 前完成经过验证的字段合并，因此变量 ID 已按规范化后的字典重新生成：

1. 血气葡萄糖统一为 mmol/L，`mg/dL ÷ {MG_DL_PER_MMOL_L_GLUCOSE:.4f}` 后与原 mmol/L 字段合并。
2. 动脉/肺泡氧分压比统一为百分比；倍数表示字段乘以 100 后合并。原始公式核验分别与 `100×PO₂/平均肺泡PO₂` 和 `PO₂/肺泡PO₂` 高度一致。
3. 患者条件 P50 合并“患者体温下 P50”“P50”和“血氧浓度 P50”三个设备别名；同时间值分别有 99.98% 和 97.84% 完全相同。
4. 固定 pH 7.4、PCO₂ 40 mmHg、37℃ 定义的字段合并为标准条件 P50。标准条件 P50 与患者条件 P50 保持为两个不同指标。

原先血气葡萄糖和氧比值的相反方向来自不同检验套覆盖了不同患者时间窗口，不是数值换算后仍然相反。规范化规则、影响行数和换算前后范围见 `../tables/variable_harmonization_audit.csv`；公式、同时间一致性和近邻换算证据见 `../tables/field_equivalence_evidence.csv`。

首末变化先在每次住院内计算；同一患者存在多次住院时，再取患者内变化中位数，确保推断统计中每名患者只贡献一次。使用双侧 Wilcoxon 符号秩检验，并在全部变量间进行 Benjamini-Hochberg FDR 校正。置信区间为患者级 bootstrap 中位数 95% CI。Theil-Sen 斜率用于描述每个住院 episode 内的稳健日变化。

## 手术对齐补充分析

- 有有效 CABG 时间的源表住院：{surgery_summary['valid_cabg_episodes']} 次
- 可关联清洗后化验的 CABG 住院：{surgery_summary['linked_cabg_episodes']} 次，{surgery_summary['linked_cabg_patients']} 人
- 可关联清洗后化验的全部有效主手术住院：{surgery_summary['linked_all_surgery_episodes']} 次
- CABG 手术时长中位数：{surgery_summary['cabg_duration_hours']['median']:.2f} 小时
- 进入 CABG 手术配对推断的“检验项×对比”：{surgery_summary['cabg_tested_contrasts']} 个，其中 BH-FDR q<0.05：{surgery_summary['cabg_fdr_significant_contrasts']} 个

手术名称、开始时间和结束时间中的多值字段按 `^` 分割并严格按位置配对。主要队列以 CABG 时间为锚点，全部有效主手术作为敏感性分析。详细的分期定义、覆盖、配对结果、审计表和图表见 `SURGERY_REPORT.md`。

## 主要增加项

| ID | 检验项 | 单位 | 中位首末变化 | 变化/IQR | BH q |
|---|---|---:|---:|---:|---:|
{rows(increases)}

## 主要降低项

| ID | 检验项 | 单位 | 中位首末变化 | 变化/IQR | BH q |
|---|---|---:|---:|---:|---:|
{rows(decreases)}

## 重要解释限制

1. “增加”或“降低”只表示数值方向，不自动代表临床改善或恶化；不同项目的正常区间和治疗目标不同。
2. 本实验为观察性住院轨迹描述，采样时间由临床需要决定，不能解释为治疗因果效应。
3. `<`、`>` 等截尾结果被排除，避免把检测限当成真实值；相关数量保存在机器可读审计中。
4. 只有 `../tables/variable_harmonization_audit.csv` 中列出的单位换算和别名会合并；其余语义近似但未经验证的检验项保持独立。
5. 住院进程图对每名患者每个时间分箱先取中位数，因此不是原始测量行数加权。
6. PDF 中 `Episode first/last values` 的箱线图使用全部 episode 配对，连线为至多 80 个 episode 的固定随机抽样；它是 episode 级描述图，而报告中的推断仍以患者为统计单位。
7. 字段合并只应用于有公式、同时间一致性或明确同义定义支持的项目；患者条件 P50 和标准条件 P50 未因名称相近而互相合并。

## 图表

- `../figures/cohort_flow.png`：数据纳入流程
- `../figures/analyte_coverage.png`：检验项覆盖与重复测量 episode
- `../figures/measurement_timing.png`：住院时间中的采样分布
- `../figures/paired_change_forest.png`：患者级首末变化及 bootstrap CI
- `../figures/change_significance.png`：效应大小与 FDR
- `../figures/longitudinal_heatmap.png`：标准化住院进程轨迹
- `../figures/top_trajectory_panels.png`：变化最大的 12 个项目
- `../figures/per_analyte/*.png`：每个纵向变量的完整分项图
- `../figures/all_analyte_trajectories.pdf`：所有分项图的多页 PDF
- `../figures/surgery_cohort_overview.png`：手术队列、术式和 CABG 时间概览
- `../figures/surgery_phase_coverage.png`：手术相对阶段覆盖
- `../figures/cabg_surgery_aligned_heatmap.png`：CABG 对齐轨迹热图
- `../figures/cabg_surgery_trajectory_panels.png`：CABG 变化最大的分项轨迹
- `../figures/cabg_surgery_contrast_forest.png`：CABG 配对变化及 bootstrap CI
- `../figures/surgery_all_vs_cabg_sensitivity.png`：全部手术与 CABG 敏感性比较
- `../figures/all_surgery_aligned_trajectories.pdf`：全部手术分项轨迹 PDF

## 机器可读结果

- `../tables/analyte_dictionary.csv`
- `../tables/analyte_coverage.csv`
- `../tables/cleaned_longitudinal_measurements.csv`
- `../tables/episode_level_changes.csv`
- `../tables/patient_level_changes.csv`
- `../tables/paired_change_statistics.csv`
- `../tables/episode_theil_sen_slopes.csv`
- `../tables/slope_statistics.csv`
- `../tables/binned_trajectories.csv`
- `../tables/surgery_event_audit.csv`
- `../tables/surgery_episode_audit.csv`
- `../tables/surgery_procedure_summary.csv`
- `../tables/surgery_phase_summary.csv`
- `../tables/surgery_contrast_statistics.csv`
- `../tables/variable_harmonization_audit.csv`
- `../tables/field_equivalence_evidence.csv`
- `../metadata/data_quality_report.json`
- `../metadata/analysis_manifest.json`
- `../metadata/surgery_analysis_manifest.json`
"""
    with open(os.path.join(output_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(text)


def run(source_path=LAB_CSV, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_dirs = _output_directories(output_dir)
    figure_dir = output_dirs["figures"]
    table_dir = output_dirs["tables"]
    report_dir = output_dirs["reports"]
    metadata_dir = output_dirs["metadata"]
    _figure_style()
    print(f"Loading laboratory table: {source_path}", flush=True)
    (
        raw,
        measurements,
        dictionary,
        flow,
        exclusions,
        harmonization_audit,
        equivalence_evidence,
    ) = _load_and_clean(source_path)
    coverage = _coverage(measurements, dictionary)
    eligible_ids = set(
        coverage.loc[
            coverage["eligible_longitudinal"], "variable_id"
        ].astype(str)
    )
    print(
        f"Cleaned rows={len(measurements)} episodes={measurements.episode_id.nunique()} "
        f"patients={measurements.hospital_id.nunique()} "
        f"eligible_variables={len(eligible_ids)}",
        flush=True,
    )
    changes, slopes = _episode_changes(measurements, eligible_ids)
    patient_changes, change_statistics = _change_statistics(
        changes, measurements, dictionary
    )
    change_statistics = change_statistics.reset_index(drop=True)
    eligible_inference_ids = set(change_statistics["variable_id"].astype(str))
    changes = changes[changes["variable_id"].isin(eligible_inference_ids)].copy()
    slopes = slopes[slopes["variable_id"].isin(eligible_inference_ids)].copy()
    patient_changes = patient_changes[
        patient_changes["variable_id"].isin(eligible_inference_ids)
    ].copy()
    slope_statistics = _slope_statistics(
        slopes, measurements, dictionary
    )
    trajectories = _binned_trajectories(
        measurements, eligible_inference_ids, dictionary
    )
    from study.exp2_lab_longitudinal_statistics.surgery_analysis import (
        run_surgery_analysis,
    )

    print("Running surgery-aligned analysis", flush=True)
    surgery_summary = run_surgery_analysis(
        raw,
        measurements,
        dictionary,
        output_dir,
        MIN_PAIRED_PATIENTS,
        BOOTSTRAP_REPLICATES,
        SEED,
    )

    flow.to_csv(os.path.join(table_dir, "cohort_flow.csv"), index=False)
    dictionary.to_csv(
        os.path.join(table_dir, "analyte_dictionary.csv"), index=False
    )
    coverage.to_csv(
        os.path.join(table_dir, "analyte_coverage.csv"), index=False
    )
    measurements.to_csv(
        os.path.join(table_dir, "cleaned_longitudinal_measurements.csv"),
        index=False,
    )
    changes.to_csv(
        os.path.join(table_dir, "episode_level_changes.csv"), index=False
    )
    patient_changes.to_csv(
        os.path.join(table_dir, "patient_level_changes.csv"), index=False
    )
    change_statistics.to_csv(
        os.path.join(table_dir, "paired_change_statistics.csv"), index=False
    )
    slopes.to_csv(
        os.path.join(table_dir, "episode_theil_sen_slopes.csv"), index=False
    )
    slope_statistics.to_csv(
        os.path.join(table_dir, "slope_statistics.csv"), index=False
    )
    trajectories.to_csv(
        os.path.join(table_dir, "binned_trajectories.csv"), index=False
    )
    harmonization_audit.to_csv(
        os.path.join(table_dir, "variable_harmonization_audit.csv"),
        index=False,
    )
    equivalence_evidence.to_csv(
        os.path.join(table_dir, "field_equivalence_evidence.csv"),
        index=False,
    )

    _plot_cohort_flow(flow, figure_dir)
    _plot_coverage(coverage, figure_dir)
    _plot_timing(measurements, figure_dir)
    _plot_change_forest(change_statistics, figure_dir)
    _plot_significance(change_statistics, figure_dir)
    _plot_heatmap(trajectories, change_statistics, figure_dir)
    _plot_top_trajectories(trajectories, change_statistics, figure_dir)
    plot_manifest = _plot_per_analyte(
        measurements,
        changes,
        slopes,
        trajectories,
        change_statistics,
        figure_dir,
    )
    plot_manifest["png_path"] = plot_manifest["png_path"].map(
        lambda path: os.path.join("figures", path)
    )
    plot_manifest["pdf_path"] = plot_manifest["pdf_path"].map(
        lambda path: os.path.join("figures", path)
    )
    plot_manifest.to_csv(
        os.path.join(table_dir, "plot_manifest.csv"), index=False
    )

    quality = {
        "schema_version": 1,
        "source": {
            "path": os.path.abspath(source_path),
            "size_bytes": os.path.getsize(source_path),
            "sha256": _sha256(source_path),
            "raw_rows": int(len(raw)),
        },
        "episode_definition": (
            "normalized hospital patient ID + admission time + discharge time"
        ),
        "variable_definition": (
            "verified harmonization rules, then canonical test name + unit"
        ),
        "inclusion": {
            "report_time": "within closed admission-discharge interval",
            "result": "numeric and not prefixed by a censoring comparator",
            "same_timestamp_duplicates": "median",
        },
        "counts": {
            "patients": int(measurements["hospital_id"].nunique()),
            "episodes": int(measurements["episode_id"].nunique()),
            "collapsed_measurements": int(len(measurements)),
            "analyte_unit_variables": int(len(dictionary)),
            "longitudinal_eligible_variables": int(len(eligible_ids)),
            "inference_variables": int(len(change_statistics)),
            "fdr_significant_variables": int(
                change_statistics["fdr_significant_0_05"].sum()
            ),
        },
        "exclusions": exclusions,
        "harmonization": {
            "rules_applied": int(len(harmonization_audit)),
            "rows_covered_by_rules": int(
                harmonization_audit["rows_affected"].sum()
            ),
            "audit_file": "tables/variable_harmonization_audit.csv",
            "evidence_file": "tables/field_equivalence_evidence.csv",
        },
        "surgery_analysis": surgery_summary,
    }
    with open(
        os.path.join(metadata_dir, "data_quality_report.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": 1,
        "experiment": "exp2_inpatient_lab_longitudinal_statistics",
        "seed": SEED,
        "minimum_repeated_episodes": MIN_REPEATED_EPISODES,
        "minimum_paired_patients": MIN_PAIRED_PATIENTS,
        "progress_bins": PROGRESS_BIN_COUNT,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "statistics": {
            "first_last_test": "two-sided Wilcoxon signed-rank on one median change per patient",
            "multiple_testing": "Benjamini-Hochberg FDR across variables",
            "effect_size": "rank-biserial correlation",
            "confidence_interval": "patient-level nonparametric bootstrap median 95% CI",
            "within_episode_slope": "Theil-Sen slope per hospital day",
            "trajectory_aggregation": (
                "median within patient-progress-bin, then median/IQR across patients"
            ),
            "surgery_analysis": (
                "CABG-aligned primary analysis with all valid principal "
                "surgeries as sensitivity cohort"
            ),
            "analyte_harmonization": (
                "verified unit and alias rules applied before variable IDs "
                "and same-timestamp collapse"
            ),
        },
        "harmonization_audit": "tables/variable_harmonization_audit.csv",
        "field_equivalence_evidence": "tables/field_equivalence_evidence.csv",
        "quality_report": "metadata/data_quality_report.json",
        "plot_manifest": "tables/plot_manifest.csv",
    }
    with open(
        os.path.join(metadata_dir, "analysis_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    _write_report(
        report_dir,
        source_path,
        measurements,
        dictionary,
        coverage,
        changes,
        change_statistics,
        slope_statistics,
        exclusions,
        surgery_summary,
        harmonization_audit,
        equivalence_evidence,
    )
    _clean_legacy_output_layout(output_dir)
    print(
        f"Done: inference_variables={len(change_statistics)} "
        f"FDR_significant={int(change_statistics.fdr_significant_0_05.sum())} "
        f"output={output_dir}",
        flush=True,
    )
    return change_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=LAB_CSV)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.source, args.output_dir)


if __name__ == "__main__":
    main()
