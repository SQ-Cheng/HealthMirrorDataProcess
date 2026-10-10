"""Describe matched pairs and unique lab events lost by a 12h limit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.common.time_alignment import local_naive_to_unix
from study.exp2_face_pretrained_head32_regression.config import TARGETS
from study.exp2_face_history_head32_regression.source_data import _nearest_measurement, TARGET_ANALYTES


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = STUDY / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
PROTOCOL = STUDY / "common/outputs/face_main_24h_frame_loss/protocol.json"
SEED = 20261008
BOOTSTRAPS = 2000


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_interval_distance(timestamp,start,end):
    return np.maximum(np.maximum(start-timestamp,timestamp-end),0)


def load_and_verify():
    protocol=json.loads(PROTOCOL.read_text())
    if protocol["lab_table_sha256"]!=sha256(STUDY.parent/"merged_lab_tests.csv"):
        raise RuntimeError("The main cohort is not based on the current lab table")
    base=pd.read_csv(SOURCE/"source_data/base_manifest.csv",dtype={"hospital_id":str,"video_id":str})
    labs=pd.read_csv(SOURCE/"source_data/lab_timeseries.csv",dtype={"hospital_id":str})
    quality=json.loads((SOURCE/"source_data/data_quality_report.json").read_text())
    if sha256(SOURCE/"source_data/lab_timeseries.csv")!=quality["source_fingerprints"]["lab_timeseries_cache"]["sha256"]:
        raise RuntimeError("Canonical assay cache changed")
    if labs.duplicated(["hospital_id","analyte","timestamp_unix"]).any():
        raise RuntimeError("Independent lab-event identity is not unique")
    lookup={key:list(zip(group.timestamp_unix,group.value)) for key,group in labs.groupby(["hospital_id","analyte"])}
    context=base[["hospital_id","video_id","capture_start_unix","capture_end_unix","admission_unix","discharge_unix","sex"]]
    frames=[];checks=[]
    for target in TARGETS:
        path=SOURCE/f"task_records/{target}.csv"
        if sha256(path)!=protocol["source_records_sha256"][target]:raise RuntimeError(f"Task records changed: {target}")
        records=pd.read_csv(path,dtype={"hospital_id":str,"video_id":str})
        records=records.merge(context,on=["hospital_id","video_id"],validate="one_to_one")
        records.insert(0,"target",target)
        records["retention"]=np.where(records.match_delta_h.le(12),"retained_12h","lost_12_to_24h")
        cadence=[];within24=[];within12=[]
        for row in records.itertuples(index=False):
            events=[(time,value) for time,value in lookup.get((row.hospital_id,TARGET_ANALYTES[target]),[])
                    if row.admission_unix<=time<=row.discharge_unix]
            selected24=_nearest_measurement(events,row.capture_start_unix,row.capture_end_unix,24)
            selected12=_nearest_measurement(events,row.capture_start_unix,row.capture_end_unix,12)
            if selected24 is None or selected24["timestamp_unix"]!=row.label_time_unix or not np.isclose(selected24["value"],row.raw_value,rtol=0,atol=1e-9):
                raise AssertionError(f"Saved 24h nearest lab cannot be reproduced: {target}/{row.video_id}")
            if row.retention=="retained_12h":
                if selected12 is None or selected12["timestamp_unix"]!=selected24["timestamp_unix"]:
                    raise AssertionError("A retained nearest event changed under the tighter window")
            elif selected12 is not None:
                raise AssertionError("An allegedly lost pair still has a valid 12h assay")
            times=np.sort(np.array([time for time,_ in events]))
            gaps=np.diff(times)/3600
            cadence.append(float(np.median(gaps)) if len(gaps) else np.nan)
            distances=nearest_interval_distance(times,row.capture_start_unix,row.capture_end_unix)/3600
            within24.append(int(np.sum(distances<=24)))
            within12.append(int(np.sum(distances<=12)))
        records["episode_median_assay_gap_h"]=cadence
        records["n_lab_events_within24h"]=within24
        records["n_lab_events_within12h"]=within12
        frames.append(records)
        checks.append({"target":target,"verified_pairs":len(records),"retained_same_nearest":int(records.retention.eq("retained_12h").sum()),
                       "lost_with_no_12h_candidate":int(records.retention.eq("lost_12_to_24h").sum()),"passed":True})
    pairs=pd.concat(frames,ignore_index=True)
    if pairs.groupby(["target","clinical_event_id"])[["hospital_id","raw_value","binary_label","label_time_unix","admission_unix","discharge_unix","score_threshold","sex"]].nunique().gt(1).any().any():
        raise RuntimeError("One independent assay has inconsistent identity or labels")
    if not pairs.match_delta_h.between(0,24+1e-9).all():raise RuntimeError("Out-of-window main matches")
    if pairs.groupby(["target","hospital_id"]).split.nunique().gt(1).any():raise RuntimeError("Patient split leakage")
    return pairs,pd.DataFrame(checks),protocol


def cabg_metadata(lab_hash):
    cache=STUDY/"exp4/outputs"
    manifest=cache/"experiment_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["source"]["sha256"]==lab_hash:
        episodes=pd.read_csv(cache/"surgical_episodes.csv",dtype={"hospital_id":str})
        reused=True
    else:
        from study.exp4.build_dataset import load_surgical_episodes
        episodes,_=load_surgical_episodes();reused=False
    for name in ("admission_time","discharge_time","index_surgery_start","index_surgery_end"):
        episodes[name.removesuffix("_time")+"_unix" if name.endswith("_time") else name+"_unix"] = episodes[name].map(local_naive_to_unix)
    keys=["hospital_id","admission_unix","discharge_unix"]
    if episodes.duplicated(keys).any():raise RuntimeError("CABG episode metadata is ambiguous")
    columns=keys+["index_surgery_start_unix","index_surgery_end_unix","index_surgery_name"]
    return episodes[columns],reused


def enrich_timeline(pairs,episodes):
    result=pairs.merge(episodes,on=["hospital_id","admission_unix","discharge_unix"],how="left",validate="many_to_one")
    result["cabg_time_available"]=result.index_surgery_end_unix.notna().astype(int)
    result["length_of_stay_days"]=(result.discharge_unix-result.admission_unix)/86400
    result["video_midpoint_unix"]=(result.capture_start_unix+result.capture_end_unix)/2
    for entity,column in (("lab","label_time_unix"),("video","video_midpoint_unix")):
        time=result[column]
        result[f"{entity}_days_since_admission"]=(time-result.admission_unix)/86400
        result[f"{entity}_days_until_discharge"]=(result.discharge_unix-time)/86400
        result[f"{entity}_stay_fraction"]=(time-result.admission_unix)/(result.discharge_unix-result.admission_unix)
        result[f"{entity}_hours_after_cabg_end"]=(time-result.index_surgery_end_unix)/3600
        result[f"{entity}_distance_to_cabg_h"]=nearest_interval_distance(time,result.index_surgery_start_unix,result.index_surgery_end_unix)/3600
        known=result.cabg_time_available.eq(1)
        distances=np.column_stack([result[f"{entity}_days_since_admission"]*24,result[f"{entity}_distance_to_cabg_h"],result[f"{entity}_days_until_discharge"]*24])
        nearest=np.full(len(result),"CABG unavailable",dtype=object)
        if known.any():
            values=distances[known.to_numpy()]
            labels=np.array(["Admission","CABG","Discharge"])[values.argmin(axis=1)]
            tied=np.isclose(values,values.min(axis=1)[:,None],rtol=0,atol=1e-9).sum(axis=1)>1
            labels=np.where(tied,"Tie",labels)
            nearest[known.to_numpy()]=labels
        result[f"{entity}_nearest_milestone"]=nearest
        phase=np.full(len(result),"CABG unavailable",dtype=object)
        phase[known & time.lt(result.index_surgery_start_unix)]="Pre-CABG"
        phase[known & time.ge(result.index_surgery_start_unix) & time.lt(result.index_surgery_end_unix)]="Intra-CABG"
        elapsed=result[f"{entity}_hours_after_cabg_end"]
        for lower,upper,label in ((0,24,"Postop 0-1d"),(24,72,"Postop 1-3d"),(72,168,"Postop 3-7d"),(168,np.inf,"Postop >7d")):
            phase[known & elapsed.ge(lower) & elapsed.lt(upper)]=label
        result[f"{entity}_cabg_phase"]=phase
    result["match_direction"]=np.select([result.match_signed_delta_h.lt(0),result.match_signed_delta_h.gt(0)],["Lab before video","Lab after video"],default="Lab inside video")
    if not result.lab_stay_fraction.between(-1e-9,1+1e-9).all():raise RuntimeError("Lab lies outside the matched admission")
    return result


def unique_events(pairs):
    rows=[]
    for (target,event),group in pairs.groupby(["target","clinical_event_id"]):
        retained=group.retention.eq("retained_12h").sum();lost=len(group)-retained
        status="shared" if retained and lost else "retained_only" if retained else "lost_only"
        row=group.iloc[0].to_dict()
        row.update(event_status=status,matched_video_pairs=len(group),retained_video_pairs=int(retained),lost_video_pairs=int(lost),
                   mirrors="|".join(sorted(group.mirror.unique())))
        for name in ("video_id","source_sample_id","split","mirror","retention","match_delta_h","match_signed_delta_h","match_direction"):
            row.pop(name,None)
        for name in list(row):
            if name.startswith("video_") or name.startswith("capture_") or name.startswith("n_lab_events_within") or name=="lab_patient_id":row.pop(name,None)
        rows.append(row)
    return pd.DataFrame(rows)


def patient_summary(pairs):
    variables=["raw_value","lab_stay_fraction","video_stay_fraction","lab_days_since_admission","lab_days_until_discharge",
               "lab_hours_after_cabg_end","lab_distance_to_cabg_h","length_of_stay_days","episode_median_assay_gap_h"]
    aggregation={name:(name,"median") for name in variables}
    aggregation.update(abnormal_fraction=("binary_label","mean"),pairs=("video_id","size"),cabg_available_fraction=("cabg_time_available","mean"))
    return pairs.groupby(["target","hospital_id","retention"]).agg(**aggregation).reset_index()


def bootstrap_mean(values,rng):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if len(values)<2:return np.nan,np.nan
    samples=values[rng.integers(0,len(values),size=(BOOTSTRAPS,len(values)))].mean(axis=1)
    return tuple(np.quantile(samples,[.025,.975]))


def summaries(pairs,events,patients):
    rows=[];rng=np.random.default_rng(SEED)
    for target in TARGETS:
        group=pairs.loc[pairs.target.eq(target)]
        lost=group.loc[group.retention.eq("lost_12_to_24h")];keep=group.loc[group.retention.eq("retained_12h")]
        lp=set(lost.hospital_id);kp=set(keep.hospital_id);event=events.loc[events.target.eq(target)]
        row={"target":target,"pairs24h":len(group),"pairs12h":len(keep),"lost_pairs":len(lost),"lost_fraction":len(lost)/len(group),
             "patients24h":group.hospital_id.nunique(),"patients12h":len(kp),"patients_with_lost_pairs":len(lp),"patients_lost_entirely":len(lp-kp),
             "unique_events24h":len(event),"events_lost_entirely":int(event.event_status.eq("lost_only").sum()),"events_shared":int(event.event_status.eq("shared").sum()),
             "lost_abnormal_fraction":lost.binary_label.mean(),"retained_abnormal_fraction":keep.binary_label.mean()}
        p=patients.loc[patients.target.eq(target)]
        for subset,label in ((lost,"lost"),(keep,"retained")):
            pp=p.loc[p.retention.eq("lost_12_to_24h" if label=="lost" else "retained_12h")]
            low,high=bootstrap_mean(pp.abnormal_fraction,rng)
            row[f"{label}_patient_mean_abnormal_fraction"]=pp.abnormal_fraction.mean()
            row[f"{label}_patient_mean_abnormal_ci95_low"]=low
            row[f"{label}_patient_mean_abnormal_ci95_high"]=high
            row[f"{label}_normal_pairs"]=int(subset.binary_label.eq(0).sum())
            row[f"{label}_abnormal_pairs"]=int(subset.binary_label.eq(1).sum())
            row[f"{label}_cabg_missing_fraction"]=1-subset.cabg_time_available.mean()
            for variable in ("raw_value","lab_stay_fraction","video_stay_fraction","lab_days_since_admission","lab_days_until_discharge",
                             "lab_distance_to_cabg_h","lab_hours_after_cabg_end","length_of_stay_days","episode_median_assay_gap_h"):
                for quantile,name in ((.25,"q25"),(.5,"median"),(.75,"q75")):
                    row[f"{label}_{variable}_{name}"]=subset[variable].quantile(quantile)
        pivot=p.pivot(index="hospital_id",columns="retention",values="abnormal_fraction").dropna()
        row["paired_patients_with_both"]=len(pivot)
        if len(pivot):
            difference=pivot.lost_12_to_24h-pivot.retained_12h
            row["within_patient_abnormal_fraction_difference"]=difference.mean()
            row["within_patient_difference_ci95_low"],row["within_patient_difference_ci95_high"]=bootstrap_mean(difference,rng)
        rows.append(row)
    summary=pd.DataFrame(rows)
    phase=[];mirror=[];split=[];direction=[];sex=[]
    for (target,retention),group in pairs.groupby(["target","retention"]):
        for entity in ("lab","video"):
            for name in ("nearest_milestone","cabg_phase"):
                for category,count in group[f"{entity}_{name}"].value_counts().items():
                    phase.append({"target":target,"retention":retention,"entity":entity,"variable":name,"category":category,"pairs":int(count),"fraction":count/len(group)})
        for category,count in group.match_direction.value_counts().items():direction.append({"target":target,"retention":retention,"direction":category,"pairs":int(count),"fraction":count/len(group)})
        for category,count in group.sex.value_counts(dropna=False).items():sex.append({"target":target,"retention":retention,"sex":category,"pairs":int(count),"fraction":count/len(group)})
    for (target,mirror_id),group in pairs.groupby(["target","mirror"]):
        loss=group.loc[group.retention.eq("lost_12_to_24h")]
        mirror.append({"target":target,"mirror":mirror_id,"pairs24h":len(group),"lost_pairs":len(loss),"lost_fraction":len(loss)/len(group),
                       "patients24h":group.hospital_id.nunique(),"patients_with_lost_pairs":loss.hospital_id.nunique()})
    for (target,set_name),group in pairs.groupby(["target","split"]):
        loss=group.loc[group.retention.eq("lost_12_to_24h")];keep=group.loc[group.retention.eq("retained_12h")]
        split.append({"target":target,"split":set_name,"pairs24h":len(group),"lost_pairs":len(loss),"lost_fraction":len(loss)/len(group),
                      "patients24h":group.hospital_id.nunique(),"patients12h":keep.hospital_id.nunique(),"retained_abnormal_fraction":keep.binary_label.mean()})
    return summary,pd.DataFrame(phase),pd.DataFrame(mirror),pd.DataFrame(split),pd.DataFrame(direction),pd.DataFrame(sex)


def write_report(output):
    from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS
    from study.exp2_face_pretrained_head32_regression.config import DATA_ROOT
    from study.common.face_video import raw_root
    summary=pd.read_csv(output/"attrition_summary.csv")
    pairs=pd.read_csv(output/"matched_pairs.csv",dtype={"hospital_id":str,"video_id":str})
    events=pd.read_csv(output/"unique_lab_events.csv",dtype={"hospital_id":str})
    manifest=json.loads((output/"analysis_manifest.json").read_text())
    mirrors=pd.read_csv(output/"mirror_attrition.csv").groupby("mirror").agg(pairs24h=("pairs24h","sum"),lost_pairs=("lost_pairs","sum"))
    native={path.name.removesuffix("_data") for path in raw_root().glob("mirror*_data")}
    legacy={path.name.removesuffix("_data") for path in Path(DATA_ROOT).glob("mirror*_data")}
    inventory=[]
    for mirror in sorted(native|legacy|set(mirrors.index)):
        inventory.append({"mirror":mirror,"native_raw_directory_present":mirror in native,"legacy_directory_present":mirror in legacy,
                          "matched_pairs24h":int(mirrors.loc[mirror,"pairs24h"]) if mirror in mirrors.index else 0})
    pd.DataFrame(inventory).to_csv(output/"mirror_inventory.csv",index=False)
    overall=manifest["overall"]
    lines=["# Matching-window attrition: 24h to 12h","",
           "## Scope and counts","",
           "Current native224/20frame, eight-target main cohort; fixed source data and patient splits. All nearest matches were independently reproduced.","",
           f"- Matched video-analyte pairs: {overall['pairs24h']:,}; lost: {overall['lost_pairs']:,} ({overall['lost_pairs']/overall['pairs24h']:.1%}).",
           f"- Videos retaining any target: {overall['videos24h_any_target']:,} -> {overall['videos12h_any_target']:,}; {overall['videos_lost_entirely']:,} lose all targets.",
           f"- Patients retaining any target: {overall['patients24h_any_target']:,} -> {overall['patients12h_any_target']:,}; {overall['patients_lost_entirely']:,} lose all targets.",
           f"- Independent assays with no retained 12h match: {events.event_status.eq('lost_only').sum():,}; shared assays retaining another match: {events.event_status.eq('shared').sum():,}.","",
           "The matching restriction does not delete records from the original laboratory table. Counts across targets are not independent patients.",
           "Laboratory timing refers to report timestamps in the provided table, not independently documented blood-collection times.","",
           "## Analyte-specific loss and normality","",
           "| Analyte | Pairs at 24h | Lost pairs | Lost fraction | Lost abnormal | Retained abnormal | Patients lost entirely |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {TASK_LABELS[row.target]} | {row.pairs24h} | {row.lost_pairs} | {row.lost_fraction:.1%} | {row.lost_abnormal_fraction:.1%} | {row.retained_abnormal_fraction:.1%} | {row.patients_lost_entirely} |")
    lines.extend(["","Normality uses the saved clinical labels, including sex-dependent Hb thresholds. Patient-weighted and within-patient comparisons are in the CSV, rather than claiming independent repeated videos.","",
                  "## Clinical timing","",
                  "Percentages below pool video-analyte pairs. The separate assay-level and unique-video phase tables remove those duplicate matches.","",
                  "| Subset | Lab nearest CABG | Lab nearest discharge | Lab nearest admission | Intra-CABG lab | Pre-CABG video | Assay after video |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for retention in ("lost_12_to_24h","retained_12h"):
        group=pairs.loc[pairs.retention.eq(retention)]
        rates=[group.lab_nearest_milestone.eq("CABG").mean(),group.lab_nearest_milestone.eq("Discharge").mean(),group.lab_nearest_milestone.eq("Admission").mean(),
               group.lab_cabg_phase.eq("Intra-CABG").mean(),group.video_cabg_phase.eq("Pre-CABG").mean(),group.match_direction.eq("Lab after video").mean()]
        lines.append("| "+retention+" | "+" | ".join(f"{value:.1%}" for value in rates)+" |")
    only=events.loc[events.event_status.eq("lost_only")]
    lines.extend(["",f"Among the {len(only):,} entirely lost unique assays, {only.lab_nearest_milestone.eq('CABG').mean():.1%} are nearest CABG and {only.lab_cabg_phase.eq('Intra-CABG').mean():.1%} occur inside its recorded interval.",
                  "","The discarded matches are not uniformly late/discharge-adjacent. Blood-gas and hemoglobin loss is strongly associated with preoperative video matches to subsequent surgical-period assays. Urea/creatinine show a different, mostly postoperative pattern.","",
                  "## Mirror distribution","","| Mirror | 24h pairs | Lost pairs | Lost fraction |","| --- | ---: | ---: | ---: |"])
    for name,row in mirrors.iterrows():
        lines.append(f"| {name} | {int(row.pairs24h)} | {int(row.lost_pairs)} | {row.lost_pairs/row.pairs24h:.1%} |")
    absent=[row["mirror"] for row in inventory if row["legacy_directory_present"] and not row["native_raw_directory_present"]]
    if absent:lines.extend(["",f"Legacy mirrors without a native raw-data directory at analysis time: {', '.join(absent)}. Their absence from this cohort is not loss caused by tightening 24h to 12h."])
    review=manifest["timing_review"]
    lines.extend(["","## Timing flags","",
                  f"{review['video_midpoint_in_cabg_count']} distinct videos have their midpoint inside a recorded CABG interval; {review['video_count_with_cabg_duration_above24h']} videos refer to a CABG interval longer than 24h.",
                  "These require review against the patient-held-recording assumption. They are exported in `timing_review_flags.csv`, not silently removed or used to modify training.","",
                  "## Figures","",
                  "- [Loss overview](figures/attrition_overview.png)",
                  "- [Normality and patient-weighted comparison](figures/normal_abnormal_comparison.png)",
                  "- [Independent-assay normality](figures/unique_assay_normality.png)",
                  "- [Lab nearest milestone](figures/lab_nearest_milestone.png)",
                  "- [Lab surgical phase](figures/lab_cabg_phase.png)",
                  "- [Video surgical phase](figures/video_cabg_phase.png)",
                  "- [Mirror-specific attrition](figures/mirror_attrition.png)",
                  "- [Fixed split attrition](figures/split_attrition.png)",
                  "- [Matching direction](figures/matching_direction.png)",
                  "- [Raw assay distributions](figures/raw_value_distributions.png)",
                  "- [Lab hospitalization position](figures/lab_hospital_stay_position.png)",
                  "- [Video hospitalization position](figures/video_hospital_stay_position.png)",
                  "- [Sampling cadence](figures/assay_sampling_cadence.png)","",
                  "PDF counterparts are under `figures/`. The source/definition manifest, per-pair/per-event/per-patient tables, sex distribution, split audit, and clinical phase tables are outside that directory."])
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plots-only",action="store_true");args=parser.parse_args()
    output=HERE/"outputs";output.mkdir(parents=True,exist_ok=True)
    from .plots import plot_all
    if args.plots_only:plot_all(output);write_report(output);return
    pairs,checks,protocol=load_and_verify()
    episodes,reused=cabg_metadata(protocol["lab_table_sha256"])
    pairs=enrich_timeline(pairs,episodes)
    events=unique_events(pairs);patients=patient_summary(pairs)
    summary,phase,mirror,split,direction,sex=summaries(pairs,events,patients)
    event_phase=[]
    for (target,status),group in events.groupby(["target","event_status"]):
        for variable in ("lab_nearest_milestone","lab_cabg_phase"):
            for category,count in group[variable].value_counts().items():
                event_phase.append({"target":target,"event_status":status,"variable":variable,"category":category,"events":int(count),"fraction":count/len(group)})
    videos=pairs.drop_duplicates("video_id").copy()
    kept=set(pairs.loc[pairs.retention.eq("retained_12h"),"video_id"])
    videos["video_retention"]=np.where(videos.video_id.isin(kept),"retained_at_least_one_target","lost_all_targets")
    video_phase=[]
    for status,group in videos.groupby("video_retention"):
        for variable in ("video_nearest_milestone","video_cabg_phase"):
            for category,count in group[variable].value_counts().items():
                video_phase.append({"video_retention":status,"variable":variable,"category":category,"videos":int(count),"fraction":count/len(group)})
    videos["recorded_cabg_duration_h"]=(videos.index_surgery_end_unix-videos.index_surgery_start_unix)/3600
    flagged=videos.loc[videos.video_cabg_phase.eq("Intra-CABG") | videos.recorded_cabg_duration_h.gt(24)].copy()
    flagged["review_reason"]=np.where(flagged.video_cabg_phase.eq("Intra-CABG"),"video midpoint falls inside recorded CABG interval","recorded CABG duration exceeds 24h")
    for column in ("video_midpoint_unix","index_surgery_start_unix","index_surgery_end_unix"):
        flagged[column.removesuffix("_unix")+"_local"]=pd.to_datetime(flagged[column],unit="s",utc=True).dt.tz_convert("Asia/Shanghai").astype(str)
    tables={"matched_pairs":pairs,"unique_lab_events":events,"patient_statistics":patients,"attrition_summary":summary,
            "clinical_phase_distribution":phase,"mirror_attrition":mirror,"split_attrition":split,
            "matching_direction":direction,"sex_distribution":sex,"nearest_match_verification":checks,
            "unique_event_phase_distribution":pd.DataFrame(event_phase),"unique_video_phase_distribution":pd.DataFrame(video_phase),
            "timing_review_flags":flagged}
    for name,frame in tables.items():frame.to_csv(output/f"{name}.csv",index=False)
    videos24=set(pairs.video_id);videos12=set(pairs.loc[pairs.retention.eq("retained_12h"),"video_id"])
    patients24=set(pairs.hospital_id);patients12=set(pairs.loc[pairs.retention.eq("retained_12h"),"hospital_id"])
    manifest={
        "source":str(SOURCE),"lab_table_sha256":protocol["lab_table_sha256"],"source_task_record_sha256":protocol["source_records_sha256"],
        "lab_timeseries_sha256":sha256(SOURCE/"source_data/lab_timeseries.csv"),"base_manifest_sha256":sha256(SOURCE/"source_data/base_manifest.csv"),
        "primary_population":"current 24h native224/20frame main video-target matches, fixed split; no re-split or training",
        "lost_definition":"12 < interval distance <= 24 hours; verified no same-admission candidate at <=12h",
        "retained_definition":"interval distance <=12h; original nearest measurement verified unchanged",
        "units":{"matched_pair":"target + video", "independent_lab_event":"target + hospital_id + report timestamp",
                 "lost_only_event":"no retained video match for this event", "shared_event":"the same assay loses some video matches but retains others"},
        "cabg_reference":"same-admission first valid recorded CABG start/end from existing recovery metadata; no substitution of unrelated procedures",
        "cabg_metadata_reused":reused,"cabg_metadata_unknown_policy":"remain in full cohort, explicit unavailable category; never treated as absence of surgery",
        "nearest_milestone":"distance to admission, CABG interval or discharge, evaluated separately at lab/video time; missing CABG makes 3-way nearest category unknown",
        "postoperative_phase_bins":"pre-CABG; intra-CABG; postop 0-1d,1-3d,3-7d,>7d; unknown",
        "normality":"exact existing target thresholds including sex-dependent Hb; abnormal_fraction uses saved binary_label",
        "laboratory_time_basis":"report timestamp, not blood-collection timestamp",
        "patient_weighting":"one mean abnormal fraction per patient/subset; paired differences restricted to patients appearing in both subsets",
        "bootstrap":{"unit":"patient","resamples":BOOTSTRAPS,"seed":SEED},
        "timing_review":{"video_midpoint_in_cabg_count":int(videos.video_cabg_phase.eq("Intra-CABG").sum()),
                         "video_count_with_cabg_duration_above24h":int(videos.recorded_cabg_duration_h.gt(24).sum()),
                         "action":"export flags, do not infer timestamps are wrong or alter training/cohort"},
        "overall":{"pairs24h":len(pairs),"lost_pairs":int(pairs.retention.eq("lost_12_to_24h").sum()),
                   "videos24h_any_target":len(videos24),"videos12h_any_target":len(videos12),"videos_lost_entirely":len(videos24-videos12),
                   "patients24h_any_target":len(patients24),"patients12h_any_target":len(patients12),"patients_lost_entirely":len(patients24-patients12)},
        "not_inferred":["tightening does not delete assays from the original lab table","patient may retain other videos/targets",
                        "mirror effects may reflect different cohorts/clinical phases rather than device causation","no new 12h seed search or model performance claim"],
    }
    if sha256(STUDY.parent/"merged_lab_tests.csv")!=protocol["lab_table_sha256"]:raise RuntimeError("Lab table changed during analysis")
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    plot_all(output)
    write_report(output)
    print(summary[["target","pairs24h","pairs12h","lost_pairs","lost_fraction","patients_lost_entirely","lost_abnormal_fraction","retained_abnormal_fraction"]].to_string(index=False),flush=True)
    print(json.dumps(manifest["overall"],indent=2),flush=True)
    print(f"[analysis-complete] {output}",flush=True)


if __name__=="__main__":main()
