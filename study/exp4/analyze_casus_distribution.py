"""Audit four-lab partial-CASUS targets for existing postoperative videos."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.time_alignment import local_naive_to_unix
from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id, _parse_datetime_to_unix
from study.exp2_face_history_head32_regression.source_data import _nearest_measurement
from study.exp5_face_pair_casus.config import CASUS_ANALYTES
from study.exp5_face_pair_casus.build_dataset import (
    _casus_component, _convert_value, _normalized_unit, _numeric,
)


HERE = Path(__file__).resolve().parent
LAB_TABLE = HERE.parents[1] / "merged_lab_tests.csv"
DISPLAY = {"creatinine": "Serum creatinine", "bilirubin": "Total bilirubin",
           "lactate": "Lactate", "platelets": "Platelet count"}
COLORS = ("#386CB0", "#379A86", "#D2A23B", "#CE734D", "#9A4B69")
THRESHOLDS = {
    "creatinine": {"unit": "mg/dL", "ascending_lower_bounds": [1.2, 2.3, 4.1], "grade4_above": 5.5},
    "bilirubin": {"unit": "mg/dL", "ascending_lower_bounds": [1.2, 3.6, 7.1], "grade4_above": 14.0},
    "lactate": {"unit": "mmol/L", "ascending_lower_bounds": [2.1, 4.1, 8.1], "grade4_above": 12.0},
    "platelets": {"unit": "10^9/L = 10^3/uL", "grade0_above": 120, "descending_lower_bounds": [81, 51, 21]},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labs(output):
    columns = ["首页病案号", "检验套名称", "检验项名称", "检验值(文本)", "单位", "标本名称", "报告时间"]
    names = {item for definition in CASUS_ANALYTES.values() for item in definition["items"]}
    chunks = []
    for chunk in pd.read_csv(LAB_TABLE, dtype=str, keep_default_na=False, usecols=columns, chunksize=100000):
        chunks.append(chunk.loc[chunk["检验项名称"].isin(names)])
    raw = pd.concat(chunks, ignore_index=True)
    raw["hospital_id"] = raw["首页病案号"].map(_normalize_hospital_id)
    raw["timestamp_unix"] = _parse_datetime_to_unix(raw["报告时间"])
    raw["numeric_value"] = _numeric(raw["检验值(文本)"])
    raw["censored"] = raw["检验值(文本)"].str.match(r"^\s*[<>≤≥＜＞]", na=False)
    raw["source_unit"] = [_normalized_unit(unit, value) for unit, value in zip(raw["单位"], raw["检验值(文本)"])]
    frames, audits = [], []
    for analyte, definition in CASUS_ANALYTES.items():
        selected = raw.loc[raw["检验项名称"].isin(definition["items"])].copy()
        selected["value"] = [_convert_value(analyte, value, unit) for value, unit in zip(selected.numeric_value, selected.source_unit)]
        selected["specimen_allowed"] = selected["标本名称"].isin(definition["specimens"])
        selected["unit_supported"] = selected.value.notna()
        selected["valid"] = (
            selected.hospital_id.ne("") & selected.timestamp_unix.notna() & ~selected.censored
            & selected.specimen_allowed & selected.unit_supported
            & selected.value.between(*definition["valid_range"])
        )
        audit = selected.groupby(["检验套名称", "检验项名称", "单位", "标本名称"], dropna=False).agg(
            source_rows=("valid", "size"), retained_rows=("valid", "sum"),
            unsupported_unit_rows=("unit_supported", lambda x: (~x).sum()),
            disallowed_specimen_rows=("specimen_allowed", lambda x: (~x).sum()),
        ).reset_index()
        audit.insert(0, "analyte", analyte)
        audits.append(audit)
        valid = selected.loc[selected.valid]
        conflicts = valid.groupby(["hospital_id", "timestamp_unix"]).value.nunique().gt(1).sum()
        print(f"[labs] {analyte}: accepted_rows={len(valid)} conflicting_timestamps={conflicts}", flush=True)
        retained = valid.groupby(["hospital_id", "timestamp_unix"], as_index=False).value.median()
        retained["analyte"] = analyte
        frames.append(retained)
    pd.concat(audits, ignore_index=True).to_csv(output / "lab_source_audit.csv", index=False)
    labs = pd.concat(frames, ignore_index=True)
    labs.to_csv(output / "accepted_lab_events.csv", index=False)
    return labs


def match_videos(videos, labs, hours):
    lookups = {(patient, analyte): list(zip(group.timestamp_unix, group.value))
               for (patient, analyte), group in labs.groupby(["hospital_id", "analyte"])}
    matched, audits = [], []
    for video in videos.itertuples(index=False):
        row = {"video_id": video.video_id, "hospital_id": video.hospital_id, "split": video.split,
               "postoperative_progress": video.recovery_score, "capture_start_unix": video.capture_start_unix,
               "capture_end_unix": video.capture_end_unix}
        start = local_naive_to_unix(video.index_surgery_end)
        end = local_naive_to_unix(video.discharge_time)
        missing, times = [], []
        for analyte in CASUS_ANALYTES:
            events = [(time, value) for time, value in lookups.get((video.hospital_id, analyte), []) if start <= time <= end]
            match = _nearest_measurement(events, video.capture_start_unix, video.capture_end_unix, hours)
            row[f"{analyte}_available"] = int(match is not None)
            if match is None:
                missing.append(analyte)
                continue
            row[f"{analyte}_value"] = match["value"]
            row[f"{analyte}_points"] = _casus_component(analyte, match["value"])
            row[f"{analyte}_lab_time_unix"] = match["timestamp_unix"]
            row[f"{analyte}_match_delta_h"] = match["delta_h"]
            row[f"{analyte}_signed_delta_h"] = match["signed_delta_h"]
            times.append(match["timestamp_unix"])
        row["missing_components"] = "|".join(missing)
        row["complete_score"] = int(not missing)
        if not missing:
            row["casus_score"] = sum(row[f"{analyte}_points"] for analyte in CASUS_ANALYTES)
            row["component_lab_span_h"] = (max(times) - min(times)) / 3600
            matched.append(row.copy())
        audits.append(row)
    return pd.DataFrame(matched), pd.DataFrame(audits)


def plot(records, patients, output, hours):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    counts = records.casus_score.value_counts().reindex(range(17), fill_value=0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    bars = axes[0].bar(counts.index, counts.values, color=COLORS[0], width=.8)
    for bar, count in zip(bars, counts.values):
        if count:
            axes[0].text(bar.get_x()+bar.get_width()/2, count, f"{count}\n{count/len(records):.1%}", ha="center", va="bottom", fontsize=8)
    axes[0].set(title=f"Video-level score | {len(records):,} videos", xlabel="Four-lab partial-CASUS score (0-16)", ylabel="Videos", xticks=range(17), xlim=(-.7,16.7))
    axes[0].set_ylim(0, max(counts.max()*1.22, 1))
    median_counts = patients.score_median.value_counts().reindex(np.arange(0,16.5,.5),fill_value=0)
    bars = axes[1].bar(median_counts.index, median_counts.values, width=.42, color=COLORS[1])
    for bar, count in zip(bars, median_counts.values):
        if count:
            axes[1].text(bar.get_x()+bar.get_width()/2, count, str(count), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(median_counts.max()*1.16, 1))
    axes[1].set(title=f"One median per patient | {len(patients):,} patients", xlabel="Patient median score across matched videos", ylabel="Patients", xticks=range(17), xlim=(-.7,16.7))
    for axis in axes:
        axis.spines[["top","right"]].set_visible(False)
        axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
    figure.suptitle(f"CABG postoperative native224 videos | nearest postoperative labs within {hours:g} h", fontsize=12)
    for extension in ("png","pdf"):
        figure.savefig(figures/f"casus_score_distribution.{extension}",dpi=200)
    plt.close(figure)
    figure, axes = plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    left = np.zeros(4)
    for grade, color in enumerate(COLORS):
        sizes = np.array([records[f"{name}_points"].eq(grade).mean()*100 for name in CASUS_ANALYTES])
        axes[0].barh(range(4),sizes,left=left,color=color,label=f"{grade} points")
        for j,size in enumerate(sizes):
            if size>=4:axes[0].text(left[j]+size/2,j,f"{size:.1f}%",ha="center",va="center",fontsize=8)
        left += sizes
    axes[0].set(yticks=range(4),yticklabels=list(DISPLAY.values()),xlabel="Matched videos (%)",xlim=(0,100),title="Component score contributions")
    axes[0].legend(ncol=5,fontsize=8,loc="lower center",bbox_to_anchor=(.5,1.02))
    axes[0].set_title("Component score contributions",pad=40)
    for name,color in zip(CASUS_ANALYTES,COLORS):
        values=np.sort(records[f"{name}_match_delta_h"].to_numpy())
        axes[1].step(values,np.arange(1,len(values)+1)/len(values),where="post",label=DISPLAY[name],color=color)
    axes[1].set(xlabel="Lab distance to original video interval (h)",ylabel="Cumulative fraction",xlim=(0,hours),ylim=(0,1.02),title="Matching distance for complete scores")
    axes[1].legend(fontsize=8)
    for axis in axes:axis.spines[["top","right"]].set_visible(False)
    figure.savefig(figures/"casus_components_and_matching.png",dpi=200)
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours",type=float,default=12)
    parser.add_argument("--records",type=Path,default=HERE/"outputs/records.csv")
    parser.add_argument("--output",type=Path,default=HERE/"outputs/casus_12h_distribution")
    args=parser.parse_args()
    if not 0<args.hours<=24:parser.error("hours must be in (0,24]")
    args.output.mkdir(parents=True,exist_ok=True)
    digest=sha256(LAB_TABLE)
    videos=pd.read_csv(args.records,dtype={"hospital_id":str,"video_id":str})
    manifest=json.loads((args.records.parent/"experiment_manifest.json").read_text())
    if manifest["source"]["sha256"]!=digest:raise RuntimeError("Recovery inventory uses a different lab table")
    if videos.video_id.duplicated().any():raise AssertionError("Duplicate candidate videos")
    labs=load_labs(args.output)
    records,audit=match_videos(videos,labs,args.hours)
    if records.empty:raise RuntimeError("No complete four-component scores")
    assert records.casus_score.between(0,16).all()
    assert records.groupby("hospital_id").split.nunique().le(1).all()
    patients=records.groupby("hospital_id").agg(videos=("video_id","size"),score_min=("casus_score","min"),score_median=("casus_score","median"),score_mean=("casus_score","mean"),score_max=("casus_score","max"),score_sd=("casus_score","std")).reset_index()
    records.to_csv(args.output/"video_scores.csv",index=False)
    audit.to_csv(args.output/"video_matching_audit.csv",index=False)
    patients.to_csv(args.output/"patient_scores.csv",index=False)
    distribution=records.casus_score.value_counts().reindex(range(17),fill_value=0).rename_axis("score").reset_index(name="videos")
    distribution["video_fraction"]=distribution.videos/len(records)
    distribution["patients_ever_at_score"]=[records.loc[records.casus_score.eq(score),"hospital_id"].nunique() for score in distribution.score]
    distribution.to_csv(args.output/"score_distribution.csv",index=False)
    patient_distribution = patients.score_median.value_counts().sort_index().rename_axis("median_score").reset_index(name="patients")
    patient_distribution["patient_fraction"] = patient_distribution.patients/len(patients)
    patient_distribution.to_csv(args.output/"patient_median_score_distribution.csv",index=False)
    availability=[]
    for name in CASUS_ANALYTES:
        selected=audit.loc[audit[f"{name}_available"].eq(1)]
        availability.append({"analyte":name,"available_videos":len(selected),"available_patients":selected.hospital_id.nunique(),"missing_videos":len(audit)-len(selected)})
    pd.DataFrame(availability).to_csv(args.output/"component_availability.csv",index=False)
    metadata={
        "lab_table":str(LAB_TABLE),"lab_table_sha256":digest,"video_inventory":str(args.records),"video_inventory_sha256":sha256(args.records),
        "scope":"existing Exp4 CABG postoperative videos; twenty valid native224 frames; no preoperative-face requirement",
        "matching":"nearest per-component postoperative value in same admission; interval distance then midpoint distance then timestamp",
        "matching_hours":args.hours,"missing_components":"exclude total; never impute missing as zero",
        "score":"four-lab partial-CASUS, 0-16, higher is more abnormal; not full CASUS or its daily-worst-value protocol",
        "thresholds":THRESHOLDS,"continuous_boundary_policy":"published next-grade lower bounds; final grade strictly above last upper bound; identical to existing CASUS implementation",
        "unit_conversion":{"creatinine_umol_L_to_mg_dL":"divide by 88.4","bilirubin_umol_L_to_mg_dL":"divide by 17.104","platelets":"10^9/L numerically equals 10^3/uL"},
        "source_definitions":CASUS_ANALYTES,"duplicate_events":"median of uncensored valid values per patient, timestamp and analyte",
        "patient_plot":"one median of eligible video scores per patient; no multi-video overweighting",
        "candidate_videos":len(videos),"candidate_patients":videos.hospital_id.nunique(),"complete_videos":len(records),"complete_patients":len(patients),
        "score_mean":records.casus_score.mean(),"score_sd":records.casus_score.std(),"score_quantiles":records.casus_score.quantile([0,.25,.5,.75,.9,.95,1]).to_dict(),
        "zero_score_fraction":records.casus_score.eq(0).mean(),"component_span_h_quantiles":records.component_lab_span_h.quantile([0,.25,.5,.75,.9,1]).to_dict(),
        "paper_table":"https://pmc.ncbi.nlm.nih.gov/articles/PMC4559007/",
    }
    if sha256(LAB_TABLE)!=digest:raise RuntimeError("Lab table changed during analysis")
    (args.output/"analysis_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plot(records,patients,args.output,args.hours)
    print(json.dumps({k:metadata[k] for k in ("candidate_videos","candidate_patients","complete_videos","complete_patients","score_mean","score_sd","score_quantiles","zero_score_fraction")},indent=2),flush=True)
    print(distribution.loc[distribution.videos.gt(0)].to_string(index=False),flush=True)
    print(f"[analysis-complete] {args.output}",flush=True)


if __name__=="__main__":main()
