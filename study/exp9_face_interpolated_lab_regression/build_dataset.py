"""Prepare audited CABG-phase interpolation labels without touching Exp2."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from study.common.time_alignment import local_naive_to_unix
from study.common.video_loss import DistinctLabViewBatchSampler
from study.exp2_face_history_head32_regression.source_data import TARGET_ANALYTES, _binary_label
from study.exp2_face_pretrained_head32_regression.data import _distribution_audit, _plot_split_distributions, add_patient_split, validate_source_data
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import fit_robust_target_scaler, write_target_scalers

from . import config
from .preoperative import nearest_preoperative,additional_preoperative_inventory,combined_index,patient_assignments


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_video(times,values,start,end,surgery_start,surgery_end,admission,discharge,max_hours=24):
    times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
    if len(times)!=len(values) or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times)<=0):
        raise ValueError("Interpolation events must be finite, unique and strictly time-ordered")
    if not admission<=surgery_start<surgery_end<=discharge:
        return None,"invalid_cabg_bounds"
    if not admission<=start<=end<=discharge:
        return None,"video_outside_admission"
    if end<=surgery_start:
        phase="pre";mask=(times>=admission)&(times<surgery_start)
    elif start>=surgery_end:
        phase="post";mask=(times>=surgery_end)&(times<=discharge)
    else:
        return None,"video_overlaps_cabg_interval"
    times,values=times[mask],values[mask]
    if len(times)<2:return None,f"{phase}_fewer_than_two_events"
    before=np.searchsorted(times,start,side="left")-1
    after=np.searchsorted(times,end,side="right")
    if before<0:return None,f"{phase}_no_lab_before_video"
    if after>=len(times):return None,f"{phase}_no_lab_after_video"
    left_gap=(start-times[before])/3600
    right_gap=(times[after]-end)/3600
    if left_gap>max_hours:return None,f"{phase}_before_lab_outside_{max_hours:g}h"
    if right_gap>max_hours:return None,f"{phase}_after_lab_outside_{max_hours:g}h"
    midpoint=(start+end)/2
    right=np.searchsorted(times,midpoint,side="left")
    left=right-1
    if left<0 or right>=len(times):raise AssertionError("Coverage check failed to prevent extrapolation")
    value=float(np.interp(midpoint,times,values,left=np.nan,right=np.nan))
    alpha=float((midpoint-times[left])/(times[right]-times[left]))
    if not np.isfinite(value) or not 0<=alpha<=1:raise AssertionError("Invalid interpolation result")
    if not min(values[left],values[right])-1e-9<=value<=max(values[left],values[right])+1e-9:
        raise AssertionError("Linear interpolation overshot the support values")
    return {
        "phase":phase,"label_time_unix":midpoint,"raw_value":value,"interpolation_alpha":alpha,
        "support_left_time_unix":float(times[left]),"support_right_time_unix":float(times[right]),
        "support_left_value":float(values[left]),"support_right_value":float(values[right]),
        "support_gap_h":float((times[right]-times[left])/3600),
        "coverage_before_time_unix":float(times[before]),"coverage_after_time_unix":float(times[after]),
        "coverage_before_delta_h":float(left_gap),"coverage_after_delta_h":float(right_gap),
        "phase_lab_event_count":len(times),"at_actual_lab_timestamp":int(midpoint==times[right]),
    },"retained"


def read_sources():
    protocol=json.loads(config.SOURCE_PROTOCOL.read_text())
    if sha256(config.STUDY.parent/"merged_lab_tests.csv")!=protocol["lab_table_sha256"]:
        raise RuntimeError("Exp2 reference is not based on the current lab table")
    quality=validate_source_data(config.SOURCE/"source_data",24)
    lab_path=config.SOURCE/"source_data/lab_timeseries.csv"
    if sha256(lab_path)!=quality["source_fingerprints"]["lab_timeseries_cache"]["sha256"]:
        raise RuntimeError("Canonical laboratory time series changed")
    index_path=Path(protocol["frame_index"])
    index=FrameOffsetIndex.load(index_path)
    if sha256(index_path)!=protocol["frame_index_sha256"] or set(index.video_formats)!={"ffv1"} or not _index_is_reusable(index_path.parent,index.video_ids,"20frame"):
        raise RuntimeError("Native224 frame cache is stale")
    base=pd.read_csv(config.SOURCE/"source_data/base_manifest.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
    videos=base.loc[base.video_id.isin(index.video_ids)].copy()
    labs=pd.read_csv(lab_path,dtype={"hospital_id":str},float_precision="round_trip")
    if labs.duplicated(["hospital_id","analyte","timestamp_unix"]).any():raise RuntimeError("Duplicate canonical assay events")
    cache=config.STUDY/"exp4/outputs"
    cached_manifest=cache/"experiment_manifest.json"
    if cached_manifest.exists() and json.loads(cached_manifest.read_text())["source"]["sha256"]==protocol["lab_table_sha256"]:
        episodes=pd.read_csv(cache/"surgical_episodes.csv",dtype={"hospital_id":str})
        events=pd.read_csv(cache/"surgery_event_audit.csv",dtype={"hospital_id":str})
        metadata_reused=True
    else:
        from study.exp4.build_dataset import load_surgical_episodes
        episodes,events=load_surgical_episodes();metadata_reused=False
    for source,destination in (("admission_time","admission_unix"),("discharge_time","discharge_unix"),
                               ("index_surgery_start","surgery_start_unix"),("index_surgery_end","surgery_end_unix")):
        episodes[destination]=episodes[source].map(local_naive_to_unix)
    keys=["hospital_id","admission_unix","discharge_unix"]
    if episodes.duplicated(keys).any():raise RuntimeError("Ambiguous CABG episode metadata")
    distinct_events=events.loc[events.is_cabg.eq(True)&events.valid_event.eq(True)].copy()
    for source,destination in (("admission_time","admission_unix"),("discharge_time","discharge_unix"),
                               ("surgery_start","event_start_unix"),("surgery_end","event_end_unix")):
        distinct_events[destination]=distinct_events[source].map(local_naive_to_unix)
    counts=distinct_events.drop_duplicates(keys+["event_start_unix","event_end_unix"]).groupby(keys).size().reset_index(name="valid_surgery_count")
    episodes=episodes.drop(columns="valid_surgery_count",errors="ignore").merge(counts,on=keys,validate="one_to_one")
    videos=videos.merge(episodes[keys+["surgery_start_unix","surgery_end_unix","valid_surgery_count"]],on=keys,how="left",validate="many_to_one")
    extras=additional_preoperative_inventory(base,index,episodes[keys+["surgery_start_unix","surgery_end_unix","valid_surgery_count"]])
    expanded,index_path=combined_index(index,index_path,extras)
    extras["frame_eligible"]=extras.video_id.isin(expanded.video_ids)
    config.OUTPUT.mkdir(parents=True,exist_ok=True)
    extras.to_csv(config.OUTPUT/"additional_preoperative_inventory.csv",index=False)
    extras.loc[~extras.frame_eligible].to_csv(config.OUTPUT/"additional_preoperative_frame_exclusions.csv",index=False)
    videos=pd.concat([videos,extras.loc[extras.frame_eligible]],ignore_index=True)
    if videos.video_id.duplicated().any():raise RuntimeError("Duplicate preoperative video inventory")
    protocol={**protocol,"exp9_frame_index_path":str(index_path),"exp9_frame_index_sha256":sha256(index_path)}
    index=expanded
    references={}
    for target in config.TARGETS:
        path=config.SOURCE/f"task_records/{target}.csv"
        if sha256(path)!=protocol["source_records_sha256"][target]:raise RuntimeError(f"Reference records changed: {target}")
        references[target]=pd.read_csv(path,dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
    return videos,labs,references,index,protocol,quality,metadata_reused


def build_records(videos,labs,references):
    lookups={}
    for key,group in labs.groupby(["hospital_id","analyte"]):
        group=group.sort_values("timestamp_unix")
        lookups[key]=(group.timestamp_unix.to_numpy(float),group.value.to_numpy(float))
    records={};audit=[]
    for target in config.TARGETS:
        definition=config.reference.SCORE_DEFINITIONS[target];rows=[]
        reference=references[target].set_index("video_id",verify_integrity=True)
        for video in videos.itertuples(index=False):
            entry={"target":target,"hospital_id":video.hospital_id,"video_id":video.video_id}
            audit_entry={**entry,"original_nearest_label_available":int(video.video_id in reference.index)}
            if pd.isna(video.surgery_start_unix) or pd.isna(video.surgery_end_unix):
                audit.append({**audit_entry,"status":"no_valid_cabg_metadata"});continue
            if video.valid_surgery_count!=1:
                audit.append({**audit_entry,"status":"multiple_cabg_events_in_admission"});continue
            times,values=lookups.get((video.hospital_id,TARGET_ANALYTES[target]),(np.array([]),np.array([])))
            if video.capture_end_unix<=video.surgery_start_unix:
                label,status=nearest_preoperative(times,values,video.capture_start_unix,video.capture_end_unix,
                                                  video.surgery_start_unix,video.admission_unix,video.discharge_unix)
            else:
                label,status=interpolate_video(times,values,video.capture_start_unix,video.capture_end_unix,
                                               video.surgery_start_unix,video.surgery_end_unix,video.admission_unix,video.discharge_unix,config.MAX_ENDPOINT_DISTANCE_H)
                if label is not None:label["label_source"]="post_linear_interpolation"
            audit.append({**audit_entry,"status":status,"cabg_duration_h":(video.surgery_end_unix-video.surgery_start_unix)/3600})
            if label is None:continue
            previous=reference.loc[video.video_id] if video.video_id in reference.index else None
            if label["phase"]=="post" and previous is None:raise AssertionError("Postoperative interpolation is missing from its 24h reference")
            if previous is not None and previous.hospital_id!=video.hospital_id:raise AssertionError("Reference patient identity mismatch")
            threshold=definition["threshold"]
            if isinstance(threshold,dict):threshold=threshold["male"] if video.sex=="男" else threshold["other"]
            distance=(threshold-label["raw_value"])/definition["scale"] if definition["direction"]=="low" else (label["raw_value"]-threshold)/definition["scale"]
            segment=f"{video.hospital_id}@{video.admission_unix:.17g}|{label['phase']}|{label['support_left_time_unix']:.17g}|{label['support_right_time_unix']:.17g}"
            rows.append({**entry,**label,"mirror":video.mirror,"lab_patient_id":video.lab_patient_id,"sex":video.sex,
                         "admission_unix":video.admission_unix,"discharge_unix":video.discharge_unix,
                         "capture_start_unix":video.capture_start_unix,"capture_end_unix":video.capture_end_unix,
                         "surgery_start_unix":video.surgery_start_unix,"surgery_end_unix":video.surgery_end_unix,
                         "binary_label":_binary_label(target,label["raw_value"],video.sex),"score_threshold":threshold,
                         "score_scale":definition["scale"],"standardized_distance":distance,"abnormal_score":float(np.arcsinh(distance)),
                         "source_sample_id":f"interpolated_{target}_{video.video_id}","clinical_event_id":segment,
                         "support_segment_id":segment,"exp2_nearest_raw_value":previous.raw_value if previous is not None else np.nan,
                         "exp2_nearest_lab_time_unix":previous.label_time_unix if previous is not None else np.nan,
                         "cabg_duration_review_flag":int((video.surgery_end_unix-video.surgery_start_unix)>24*3600)})
        records[target]=pd.DataFrame(rows)
        if not records[target].empty:
            existing,extension=patient_assignments(target,references[target],records[target].hospital_id)
            records[target]["exp2_split"]=records[target].hospital_id.map(existing)
            records[target]["new_patient_split"]=records[target].hospital_id.map(extension)
            records[target]["split_origin"]=np.where(records[target].exp2_split.notna(),"exp2_existing_patient","deterministic_new_patient_extension")
    return records,pd.DataFrame(audit)


def assign_split(records,target,policy):
    if policy=="balanced_search":
        result,reason,summaries,pairs,selection=add_patient_split(records,target,config.reference.SEED)
        if result is None:raise RuntimeError(f"Balanced interpolation split failed: {target}: {reason}")
        return result,summaries,pairs,selection
    result=records.copy();result["split"]=result.exp2_split
    if "new_patient_split" in result:result["split"]=result["split"].fillna(result.new_patient_split)
    if result.split.isna().any():raise RuntimeError("Unassigned new patient")
    summaries,pairs=_distribution_audit(result,target)
    selection={"target":target,"policy":"reuse exact Exp2 patient assignment on eligible subset",
               "all_distribution_pairs_pass":bool(all(pair["passed"] for pair in pairs)),"patients_are_disjoint":True}
    return result,summaries,pairs,selection


def write_preparation_report(manifest,summary,checks,audit):
    from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS
    counts=manifest["counts"]
    lines=["# Exp9 preparation report","","## Status","",
           "Training labels prepared. Formal model fitting is controlled by the launcher/monitor; inspect autostart_status.json for live queue status.","",
           "## Definition","",
           "Preoperative videos use the nearest observed pre-CABG report from the same admission, with no maximum distance or before/after coverage requirement. It may precede or follow the video, but must be strictly before CABG start. Postoperative targets remain piecewise-linear values at the original video midpoint, with strict same-phase before/after reports each within 24 hours. No interpolation crosses surgery, and no postoperative extrapolation is permitted.","",
           "Targets are raw laboratory values: observed before surgery and interpolated after surgery, not abnormality scores or CASUS. Report time is not independently documented blood-draw time.","",
           "## Cohort","",
           f"- Candidate pool: {counts['candidate_videos']:,} videos / {counts['candidate_patients']:,} patients.",
           f"- Retained for at least one analyte: {counts['videos_with_any_target']:,} videos / {counts['patients_with_any_target']:,} patients.",
           f"- Retained video-analyte labels: {counts['video_target_pairs']:,}.","",
           "Table entries for each split are videos / patients. A patient can contribute multiple videos.","",
           "| Analyte | Total videos / patients | Train | Validation | Test | Pre / post videos |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {TASK_LABELS[row.target]} | {row.videos} / {row.patients} | {row.train_videos} / {row.train_patients} | {row.val_videos} / {row.val_patients} | {row.test_videos} / {row.test_patients} | {row.pre_videos} / {row.post_videos} |")
    total=pd.concat([pd.read_csv(config.OUTPUT/f"task_records/{target}.csv",dtype={"hospital_id":str,"video_id":str}) for target in config.TARGETS],ignore_index=True)
    pre=total.loc[total.phase.eq("pre")]
    review=manifest["metadata_review"]
    pre_targets=", ".join(TASK_LABELS[target] for target in config.TARGETS if target in set(pre.target)) or "none"
    lines.extend(["",f"Preoperative matches cover {pre.video_id.nunique()} distinct videos / {pre.hospital_id.nunique()} patients. Available analytes: {pre_targets}. Labels beyond 24h are retained by design, not claimed to be contemporaneous measurements.","",
                  "## Split and controls","",
                  "Reuse each corresponding Exp2 patient's existing train/validation/test assignment. New patients lacking that indicator's reference assignment are added with deterministic 60/20/20 allocation using the fixed seed. Existing patients never move, and no new seed search is performed. Scalers are refitted from mixed training labels only.",
                  f"{int(checks.passed.sum())}/{len(checks)} raw-value/abnormal-score pairwise distribution checks pass the inherited audit limits; failed checks are retained rather than silently reshuffling patients. Clinical-sign metrics are secondary.","",
                  "Same EfficientNet-B0/head32, native224 RGB, twenty nonadjacent frames, five views, 240-image full batch, frame-level SmoothL1(beta=0.5), two-stage 40/60 training, lr 2e-4/1e-5, patience 10/12, AdamW weight decay 1e-4, cosine floor 1e-6, and compilation as Exp2.","",
                  "The twelve distinct groups per batch are observed preoperative assay events or postoperative interpolation intervals. Repeated targets sharing their observed event or both interpolation endpoints are separated within a batch. Every frame and view remains included.","",
                  "## Exclusion reasons","",
                  "Counts below are restricted to pairs that already had a nearest-value label in Exp2. Each excluded pair has one first-failing reason.","",
                  "| Reason | Video-analyte pairs |","| --- | ---: |"])
    excluded=audit.loc[audit.original_nearest_label_available.eq(1)&audit.status.ne("retained")]
    for reason,count in excluded.status.value_counts().items():lines.append(f"| {reason} | {count} |")
    lines.extend(["","## Review flags","",
                  f"{review['videos_with_cabg_duration_above24h']} eligible distinct videos / {review['video_target_pairs_flagged']} labels reference a recorded CABG duration above 24h. These are in `cabg_metadata_review.csv`; no surgery time was silently corrected. Review this metadata before deciding whether to exclude them for formal fitting.","",
                  "## Files and launch","",
                  "Source-frame packet index and ImageNet weights are reused; no duplicated video/image cache is written.",
                  "- [Task counts](task_summary.csv)","- [Phase/split counts](phase_split_counts.csv)",
                  "- [Batch audit](batch_audit.csv)","- [Split distribution audit](split_distribution_pairwise.csv)",
                  "- [Additional preoperative inventory](additional_preoperative_inventory.csv)",
                  "- [Interpolation curve examples](figures/interpolation_curve_examples.png)",
                  "- [Bracketing distances](figures/interpolation_bracketing_distances.png)",
                  "- [Nearest versus interpolated labels](figures/interpolated_vs_nearest_targets.png)","",
                  "The monitored launcher starts formal training only after the current Exp2 mains complete successfully:","",
                  "```bash","bash study/exp9_face_interpolated_lab_regression/launch_after_current_screen.sh","```"])
    (config.OUTPUT/"REPORT.md").write_text("\n".join(lines)+"\n")


def prepare(split_policy=config.DEFAULT_SPLIT_POLICY):
    output=config.OUTPUT;output.mkdir(parents=True,exist_ok=True)
    if (output/"run_index.csv").exists():raise RuntimeError("Existing Exp9 training results cannot be overwritten by preparation")
    videos,labs,references,index,protocol,quality,metadata_reused=read_sources()
    tasks,audit=build_records(videos,labs,references)
    audit.to_csv(output/"video_target_eligibility_audit.csv",index=False)
    audit.loc[audit.original_nearest_label_available.eq(1)].groupby(["target","status"]).size().reset_index(name="video_target_pairs").to_csv(output/"exclusions_among_exp2_labelled_pairs.csv",index=False)
    task_dir=output/"task_records";task_dir.mkdir(exist_ok=True)
    summary=[];phase_rows=[];distribution=[];pairs=[];selections=[];scalers={};batch_audit=[]
    for target,records in tasks.items():
        if records.empty:raise RuntimeError(f"Insufficient phase-separated laboratory data: {target}")
        records,stats,checks,selection=assign_split(records,target,split_policy)
        if set(records.split)!={"train","val","test"}:raise RuntimeError(f"Missing split: {target}")
        if records.groupby("hospital_id").split.nunique().gt(1).any() or records.video_id.duplicated().any():raise AssertionError("Patient leakage or duplicate video")
        records.to_csv(task_dir/f"{target}.csv",index=False,float_format="%.17g")
        records=pd.read_csv(task_dir/f"{target}.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
        scaler=fit_robust_target_scaler(target,records,config.reference.SCORE_DEFINITIONS[target]["unit"])
        records["robust_scaled_raw_value"]=scaler.transform(records.raw_value)
        records.to_csv(task_dir/f"{target}.csv",index=False,float_format="%.17g")
        tasks[target]=records;scalers[target]=scaler;distribution.extend(stats);pairs.extend(checks);selections.append(selection)
        row={"target":target,"status":"ready","videos":len(records),"patients":records.hospital_id.nunique(),
             "exp2_reference_videos":len(references[target]),"retained_fraction_of_exp2":len(records)/len(references[target]),
             "support_segments":records.support_segment_id.nunique(),"pre_videos":int(records.phase.eq("pre").sum()),
             "post_videos":int(records.phase.eq("post").sum()),"support_gap_h_median":records.support_gap_h.median(),
             "pre_nearest_beyond24h_videos":int((records.phase.eq("pre")&records.selected_lab_delta_h.gt(24)).sum()),
             "new_patients_without_exp2_assignment":int(records.loc[records.split_origin.eq("deterministic_new_patient_extension"),"hospital_id"].nunique()),
             "support_gap_h_p95":records.support_gap_h.quantile(.95),"positive_videos":int(records.binary_label.sum())}
        for split,group in records.groupby("split"):
            row[f"{split}_videos"]=len(group);row[f"{split}_patients"]=group.hospital_id.nunique()
            if split=="train":
                dataset=SimpleNamespace(video_records=group.reset_index(drop=True),expand_all_views=False,
                                        frame_video_rows=np.repeat(np.arange(len(group)),20),views=config.reference.VIEW_NAMES)
                sampler=DistinctLabViewBatchSampler(dataset,240);batches=list(sampler)
                np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(len(group)*100))
                for batch in batches:
                    segments=group.reset_index(drop=True).iloc[np.asarray(batch).reshape(-1,20)[:,0]//100].support_segment_id
                    if segments.nunique()!=len(segments):raise AssertionError("Repeated interpolation support segment in one batch")
                batch_audit.append({"target":target,"train_videos":len(group),"support_segments":group.support_segment_id.nunique(),
                                    "batches":len(batches),"full_240_frame_batches":sum(len(batch)==240 for batch in batches),
                                    "partial_batches":sum(len(batch)<240 for batch in batches),"epoch_inputs":len(group)*100,
                                    "distinct_support_segments_per_full_batch":12})
            for phase,phase_group in group.groupby("phase"):
                phase_rows.append({"target":target,"split":split,"phase":phase,"videos":len(phase_group),"patients":phase_group.hospital_id.nunique(),"segments":phase_group.support_segment_id.nunique()})
        summary.append(row)
    write_target_scalers(scalers,output/"target_scalers.json")
    pd.DataFrame(summary).to_csv(output/"task_summary.csv",index=False)
    pd.DataFrame(phase_rows).to_csv(output/"phase_split_counts.csv",index=False)
    pd.DataFrame(batch_audit).to_csv(output/"batch_audit.csv",index=False)
    pd.DataFrame(distribution).to_csv(output/"split_distribution_audit.csv",index=False)
    pd.DataFrame(pairs).to_csv(output/"split_distribution_pairwise.csv",index=False)
    _plot_split_distributions(tasks,output)
    (output/"split_assignment_manifest.json").write_text(json.dumps({"policy":split_policy,"reference_seed":config.reference.SEED,"target_results":selections},indent=2)+"\n")
    union=pd.concat(list(tasks.values()),ignore_index=True).drop_duplicates("video_id")
    all_records=pd.concat(list(tasks.values()),ignore_index=True)
    review=all_records.loc[all_records.cabg_duration_review_flag.eq(1)]
    review.to_csv(output/"cabg_metadata_review.csv",index=False)
    manifest={
        "experiment":"exp9_face_interpolated_lab_regression","targets":list(config.TARGETS),"split_policy":split_policy,
        "lab_table_sha256":protocol["lab_table_sha256"],"source_reference":str(config.SOURCE),
        "source_task_record_sha256":protocol["source_records_sha256"],"lab_timeseries_sha256":sha256(config.SOURCE/"source_data/lab_timeseries.csv"),
        "frame_index_path":protocol["exp9_frame_index_path"],"frame_index_sha256":protocol["exp9_frame_index_sha256"],"frame_cache_reused":True,
        "parent_frame_index_path":protocol["frame_index"],"parent_frame_index_sha256":protocol["frame_index_sha256"],
        "task_records_sha256":{target:sha256(task_dir/f"{target}.csv") for target in config.TARGETS},
        "scalers_sha256":sha256(output/"target_scalers.json"),"counts":{"candidate_videos":len(videos),"candidate_patients":int(videos.hospital_id.nunique()),
                                                                       "videos_with_any_target":len(union),"patients_with_any_target":int(union.hospital_id.nunique()),"video_target_pairs":sum(len(records) for records in tasks.values())},
        "label_policy":{"interpolation":config.INTERPOLATION,"preoperative":config.PREOPERATIVE_POLICY,"point_time":config.VIDEO_LABEL_TIME,
                        "phase":"each patient/analyte/admission separately; pre-CABG start and post-CABG end; intraoperative reports excluded",
                        "full_video_bracketing":"postoperative only: strictly earlier than capture start and strictly later than capture end in the same phase",
                        "maximum_distance_each_side_hours":config.MAX_ENDPOINT_DISTANCE_H,"preoperative_distance_limit":None,"extrapolation":False,
                        "surgery_crossing_video_policy":"excluded, including partially overlapping recordings",
                        "missing_cabg_metadata":"excluded; never substitute an unrelated surgery",
                        "multiple_cabg_events_in_admission":"excluded because one pre and one post curve cannot resolve multiple surgical discontinuities without a new protocol",
                        "laboratory_time_basis":"report timestamps, not independently recorded blood draw timestamps",
                        "long_cabg_duration":"flag for review, do not silently redefine surgery time",
                        "retrospective_pseudo_labels":"future observations define targets, not model inputs; no claim of directly measured instantaneous ground truth"},
        "training":config.training_settings(),"analyte_source_policies":quality["analyte_source_policies"],"cabg_metadata_reused":metadata_reused,
        "batch_independence":"preoperative group is one observed assay; postoperative group is one interpolation support segment; same source group never repeats in a batch",
        "comparability":"same backbone/head/optimization; original packet offsets reused; existing patients keep their original indicator-specific split and new patients receive deterministic extensions" if split_policy=="reuse_exp2" else "same pipeline and balanced-search algorithm, but independently selected patient splits",
        "metadata_review":{"videos_with_cabg_duration_above24h":int(review.video_id.nunique()),"video_target_pairs_flagged":len(review),"action":"flagged, not silently corrected or excluded; no formal training launched"},
        "training_started":False,
    }
    if sha256(config.STUDY.parent/"merged_lab_tests.csv")!=protocol["lab_table_sha256"]:raise RuntimeError("Lab table changed during preparation")
    (output/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    write_preparation_report(manifest,pd.DataFrame(summary),pd.DataFrame(pairs),audit)
    (output/"DATA_READY").write_text("phase-separated interpolation labels and audits prepared; no training started\n")
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(json.dumps(manifest["counts"],indent=2),flush=True)
    return manifest
