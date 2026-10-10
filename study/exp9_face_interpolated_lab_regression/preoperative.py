"""Unrestricted pre-CABG matching, patient-split extension and index reuse."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from study.exp2_face_history_head32_regression.source_data import _nearest_measurement
from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id
from study.exp2_lab_multimodal.build_dataset import _read_merged_patient_info
from study.common.face_video import raw_root
from study.common.time_alignment import read_video_session,TimeAlignmentError
from study.exp2_face_history_head32_regression.source_data import _load_hospital_episodes
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex,build_or_reuse_frame_index,_index_is_reusable
from . import config


def nearest_preoperative(times,values,start,end,surgery_start,admission,discharge):
    times=np.asarray(times,float);values=np.asarray(values,float)
    if len(times)!=len(values) or not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times)<=0):
        raise ValueError("Preoperative events must be finite and strictly time-ordered")
    if not admission<=start<=end<=surgery_start<=discharge:return None,"invalid_preoperative_video_bounds"
    mask=(times>=admission)&(times<surgery_start)
    times,values=times[mask],values[mask]
    nearest=_nearest_measurement(list(zip(times,values)),start,end,float("inf"))
    if nearest is None:return None,"pre_no_valid_preoperative_lab"
    time=nearest["timestamp_unix"];value=nearest["value"]
    return {
        "phase":"pre","label_source":"pre_nearest_observed","label_time_unix":(start+end)/2,
        "raw_value":value,"selected_lab_time_unix":time,"selected_lab_value":value,
        "selected_lab_delta_h":nearest["delta_h"],"selected_lab_signed_delta_h":nearest["signed_delta_h"],
        "support_left_time_unix":time,"support_right_time_unix":time,
        "support_left_value":value,"support_right_value":value,"interpolation_alpha":np.nan,
        "support_gap_h":np.nan,"coverage_before_time_unix":np.nan,"coverage_after_time_unix":np.nan,
        "coverage_before_delta_h":np.nan,"coverage_after_delta_h":np.nan,
        "phase_lab_event_count":len(times),"at_actual_lab_timestamp":int(time==(start+end)/2),
    },"retained"


def additional_preoperative_inventory(base,index,episodes):
    """Include validated sessions rejected by the original 24h lab gate."""
    audit=pd.read_csv(config.SOURCE/"source_data/raw_video_audit.csv",dtype={"hospital_id":str,"video_id":str})
    valid=audit.loc[audit.status.isin(("retained_24h_pool","supported_lab_outside_24h","patient_without_supported_lab"))].copy()
    ledger=raw_root()/"_face224_processing/index.csv"
    if ledger.exists():
        native=pd.read_csv(ledger,dtype={"video_id":str})
        only=native.loc[~native.video_id.isin(audit.video_id.dropna())].drop_duplicates("video_id")
        mapping=_read_merged_patient_info();admissions=_load_hospital_episodes();rows=[];review=[]
        for video in only.itertuples(index=False):
            import re
            parsed=re.fullmatch(r"(mirror\d+)_patient_(\d+)",video.video_id)
            if parsed is None:continue
            mirror,local=parsed.group(1),int(parsed.group(2))
            info=mapping.get((mirror,local),{})
            hospital=_normalize_hospital_id(info.get("Hospital_Patient_ID",""))
            if not hospital:
                review.append({"video_id":video.video_id,"status":"missing_or_invalid_hospital_mapping"});continue
            path=raw_root()/f"{mirror}_data/patient_{local:06d}/raw_video.avi"
            try:bounds=read_video_session(path,expected_local_id=local,expected_hospital_id=hospital)
            except TimeAlignmentError as exc:
                review.append({"video_id":video.video_id,"status":exc.code});continue
            episodes_for_video=[ep for ep in admissions.get(hospital,()) if ep[0]<=bounds["capture_start_unix"] and bounds["capture_end_unix"]<=ep[1]]
            if len(episodes_for_video)!=1:
                review.append({"video_id":video.video_id,"status":"missing_or_ambiguous_admission"});continue
            rows.append({"hospital_id":hospital,"video_id":video.video_id,**bounds})
            review.append({"video_id":video.video_id,"status":"validated_native_only_session"})
        config.OUTPUT.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(review).to_csv(config.OUTPUT/"native_only_inventory_audit.csv",index=False)
        if rows:valid=pd.concat([valid,pd.DataFrame(rows)],ignore_index=True)
    valid=valid.loc[~valid.video_id.isin(index.video_ids)]
    candidates=valid[["hospital_id","video_id","capture_start_unix","capture_end_unix"]].merge(episodes,on="hospital_id",how="inner")
    candidates=candidates.loc[candidates.capture_start_unix.ge(candidates.admission_unix)&candidates.capture_end_unix.le(candidates.discharge_unix)&
                              candidates.capture_end_unix.le(candidates.surgery_start_unix)&candidates.valid_surgery_count.eq(1)].copy()
    if candidates.video_id.duplicated().any():raise RuntimeError("Additional preoperative session belongs to multiple CABG episodes")
    parsed=candidates.video_id.str.extract(r"^(mirror\d+)_patient_(\d+)$")
    if parsed.isna().any().any():raise RuntimeError("Invalid additional video identity")
    candidates["mirror"]=parsed[0];candidates["lab_patient_id"]=parsed[1].astype(int)
    sex=pd.read_csv(config.STUDY.parent/"merged_lab_tests.csv",usecols=["首页病案号","首页性别"],dtype=str,keep_default_na=False)
    sex["hospital_id"]=sex["首页病案号"].map(_normalize_hospital_id)
    sex=sex.loc[sex["首页性别"].str.strip().ne("")].drop_duplicates("hospital_id")
    candidates["sex"]=candidates.hospital_id.map(sex.set_index("hospital_id")["首页性别"]).fillna("")
    return candidates


def combined_index(parent,parent_path,extra):
    if extra.empty:return parent,Path(parent_path)
    destination=config.CACHE/"combined_frames20";destination.mkdir(parents=True,exist_ok=True)
    output=destination/"frame_offsets.npz"
    expected=set(parent.video_ids)|set(extra.video_id)
    if _index_is_reusable(destination,expected,"20frame"):
        return FrameOffsetIndex.load(output),output
    delta_dir=config.CACHE/"additional_preoperative_frames20"
    delta=build_or_reuse_frame_index(extra,str(delta_dir),"20frame")
    arrays={name:np.concatenate([getattr(parent,name),getattr(delta,name)]) for name in
            ("video_ids","video_paths","video_formats","codec_extradata","starts","ends","source_indices")}
    arrays["video_ptr"]=np.concatenate([parent.video_ptr,delta.video_ptr[1:]+parent.video_ptr[-1]])
    if len(set(arrays["video_ids"]))!=len(arrays["video_ids"]):raise RuntimeError("Duplicate video in combined packet index")
    np.savez_compressed(str(output)+".tmp.npz",**arrays);os.replace(str(output)+".tmp.npz",output)
    original=json.loads(Path(parent_path).with_name("index_manifest.json").read_text())
    added=json.loads((delta_dir/"index_manifest.json").read_text())
    videos={row["video_id"]:row for row in original["videos"]+added["videos"]}
    failed={row["video_id"]:row for row in original["failed_videos"]+added["failed_videos"] if row["video_id"] not in videos}
    merged={**original,"videos":list(videos.values()),"failed_videos":list(failed.values()),
            "total_valid_frames":len(arrays["starts"]),"indexed_video_count":len(videos),"failed_video_count":len(failed),
            "storage_policy":"reused parent packet offsets plus newly indexed preoperative videos; no copied pixels"}
    summary=pd.concat([pd.read_csv(Path(parent_path).with_name("video_frame_summary.csv")),pd.read_csv(delta_dir/"video_frame_summary.csv")],ignore_index=True).drop_duplicates("video_id",keep="last")
    summary.to_csv(destination/"video_frame_summary.csv",index=False)
    failures=pd.concat([pd.read_csv(Path(parent_path).with_name("invalid_frames.csv")),pd.read_csv(delta_dir/"invalid_frames.csv")],ignore_index=True).drop_duplicates()
    failures.to_csv(destination/"invalid_frames.csv",index=False);merged["total_invalid_frames"]=len(failures)
    (destination/"index_manifest.json").write_text(json.dumps(merged,indent=2)+"\n")
    print(f"[frame-cache] reused={len(parent.video_ids)} additional={len(delta.video_ids)}; active Exp2 index unchanged",flush=True)
    return FrameOffsetIndex.load(output),output


def patient_assignments(target,reference,patient_ids):
    if reference.groupby("hospital_id").split.nunique().gt(1).any():raise RuntimeError("Reference patient leakage")
    existing=reference.groupby("hospital_id").split.first().to_dict()
    new=sorted(set(patient_ids)-set(existing))
    offset=int.from_bytes(hashlib.sha256(target.encode()).digest()[:4],"little")
    rng=np.random.default_rng((config.reference.SEED+offset)%(2**32))
    shuffled=np.asarray(new,dtype=object)[rng.permutation(len(new))]
    if len(new)>=3:
        train=max(1,round(.6*len(new)));val=max(1,round(.2*len(new)))
        train=min(train,len(new)-2);val=min(val,len(new)-train-1)
        labels=["train"]*train+["val"]*val+["test"]*(len(new)-train-val)
    else:labels=rng.choice(["train","val","test"],size=len(new),p=[.6,.2,.2]).tolist()
    extension=dict(zip(shuffled,labels))
    return existing,extension
