"""Check HCT/eGFR/RBC/HDL-C/UA/HbA1c/HCY without changing model data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from study.exp2_lab_multimodal.build_dataset import _normalize_hospital_id,_parse_datetime_to_unix,_extract_numeric
from study.exp2_face_history_head32_regression.source_data import _load_hospital_episodes,_nearest_measurement
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex,_index_is_reusable,build_or_reuse_frame_index


HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
SOURCE=STUDY/"exp2_face_pretrained_head32_regression/outputs/20frame_face224/source_data"
LAB=STUDY.parent/"merged_lab_tests.csv"
DEFINITIONS={
    "hct":{"cn":"红细胞压积（血细胞比容）","items":["*红细胞压积","红细胞压积"],"unit":"%","range":[1,85]},
    "egfr_cr_reported":{"cn":"估算肾小球滤过率（肌酐法报告值）","items":["eGFR(CKD-EPI 肌酐)"],"unit":"mL/min/1.73m2","range":[0,300]},
    "egfr_cys_reported":{"cn":"估算肾小球滤过率（胱抑素C法报告值）","items":["eGFR(CKD-EPI 胱抑素C)"],"unit":"mL/min/1.73m2","range":[0,300]},
    "hba1c_pct":{"cn":"糖化血红蛋白百分比","items":["*糖化血红蛋白","糖化血红蛋白"],"unit":"%","range":[2,25]},
    "rbc_direct":{"cn":"红细胞计数","items":["*红细胞","红细胞","*红细胞计数","红细胞计数","RBC"],"unit":"10^12/L","range":[.2,12]},
    "hdlc":{"cn":"高密度脂蛋白胆固醇","items":["*高密度脂蛋白胆固醇","高密度脂蛋白胆固醇","*高密度脂蛋白胆固醇(HDL-C)测定","高密度脂蛋白胆固醇(HDL-C)测定","HDL-C"],"unit":"mmol/L","range":[.05,10]},
    "ua":{"cn":"尿酸","items":["*尿酸","尿酸","*尿酸(UA)测定","尿酸(UA)测定","UA"],"unit":"umol/L","range":[1,3000]},
    "hcy":{"cn":"同型半胱氨酸","items":["*同型半胱氨酸","同型半胱氨酸","*同型半胱氨酸(Hcy)测定","同型半胱氨酸(Hcy)测定","Hcy","HCY"],"unit":"umol/L","range":[.1,500]},
    "rbc_derived":{"cn":"红细胞计数（同次血常规Hb/MCH反算）","items":[],"unit":"10^12/L","range":[.2,12]},
    "hct_cbc_derived":{"cn":"红细胞压积（同次血常规Hb/MCHC反算）","items":[],"unit":"%","range":[1,85]},
    "hct_direct_or_cbc_derived":{"cn":"红细胞压积（直接值优先，缺失时血常规反算）","items":[],"unit":"%","range":[1,85]},
    "egfr_cr_2021_derived":{"cn":"估算肾小球滤过率（CKD-EPI 2021肌酐公式计算）","items":[],"unit":"mL/min/1.73m2","range":[0,300]},
}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def egfr_2021(creatinine_umol_l,age,female):
    cr=np.asarray(creatinine_umol_l,float)/88.4
    age=np.asarray(age,float);female=np.asarray(female,bool)
    k=np.where(female,.7,.9);alpha=np.where(female,-.241,-.302)
    return 142*np.minimum(cr/k,1)**alpha*np.maximum(cr/k,1)**-1.2*.9938**age*np.where(female,1.012,1)


def normalize_unit(series):
    return series.str.lower().str.replace(r"\s+","",regex=True).str.replace("µ","u",regex=False).str.replace("μ","u",regex=False).str.replace("∧","^",regex=False)


def collapse(frame):
    return frame.groupby(["hospital_id","timestamp_unix"],as_index=False).value.median()


def load_events(output):
    columns=["首页病案号","首页性别","首页就诊时年龄","检验套名称","检验项名称","检验值(文本)","单位","标本名称","报告时间"]
    raw=pd.read_csv(LAB,dtype=str,keep_default_na=False,usecols=columns)
    raw["hospital_id"]=raw["首页病案号"].map(_normalize_hospital_id)
    raw["timestamp_unix"]=_parse_datetime_to_unix(raw["报告时间"])
    raw["numeric"]=_extract_numeric(raw["检验值(文本)"].str.replace(",","",regex=False))
    raw["unit_norm"]=normalize_unit(raw["单位"])
    raw["base_valid"]=raw.hospital_id.ne("")&raw.timestamp_unix.notna()&raw.numeric.notna()&~raw["检验值(文本)"].str.match(r"^\s*(?:[<>≤≥＜＞]|大于|小于|超过|低于)",na=False)
    raw["blood"]=raw["标本名称"].isin(("血","血清","血浆","全血","动脉血即刻","即刻动脉血","静脉血"))
    events={};source_counts={};audits=[]
    for code,definition in DEFINITIONS.items():
        if not definition["items"]:continue
        selected=raw.loc[raw["检验项名称"].isin(definition["items"])].copy()
        selected["value"]=selected.numeric
        if code=="hct":supported=selected.unit_norm.eq("%")
        elif code.startswith("egfr"):
            supported=selected.unit_norm.str.contains(r"ml/\(?min/?1[.]73(?:㎡|m2|m\^2)",regex=True)
        elif code=="hba1c_pct":
            supported=selected.unit_norm.isin(("%","mmol/mol"))
            selected.loc[selected.unit_norm.eq("mmol/mol"),"value"]=selected.numeric*.09148+2.152
        elif code=="rbc_direct":supported=selected.unit_norm.isin(("10^12/l","*10^12/l","t/l","10^6/ul"))
        elif code=="hdlc":supported=selected.unit_norm.eq("mmol/l")
        else:supported=selected.unit_norm.eq("umol/l")
        selected["valid"]=selected.base_valid&selected.blood&supported&selected.value.between(*definition["range"])
        audits.append(selected.groupby(["检验项名称","单位","标本名称"],dropna=False).agg(source_rows=("valid","size"),valid_rows=("valid","sum")).reset_index().assign(indicator=code))
        kept=selected.loc[selected.valid].copy()
        if code=="hba1c_pct":
            # Prefer directly reported percentages when both unit forms exist.
            pct_keys=pd.MultiIndex.from_frame(kept.loc[kept.unit_norm.eq("%"),["hospital_id","timestamp_unix"]])
            duplicate_ifcc=kept.unit_norm.eq("mmol/mol")&pd.MultiIndex.from_frame(kept[["hospital_id","timestamp_unix"]]).isin(pct_keys)
            kept=kept.loc[~duplicate_ifcc]
        events[code]=collapse(kept)
        source_counts[code]=len(selected)
    cbc=raw.loc[raw.base_valid&raw["标本名称"].isin(("血","全血"))&raw["检验套名称"].str.contains("血细胞|血常规",regex=True)].copy()
    keys=["hospital_id","timestamp_unix","检验套名称","标本名称"]
    hb=cbc.loc[cbc["检验项名称"].isin(("*血红蛋白","血红蛋白"))&cbc.unit_norm.isin(("g/l","g/dl"))].copy()
    hb["value"]=hb.numeric*np.where(hb.unit_norm.eq("g/dl"),10,1)
    hb=hb.loc[hb.value.between(20,250)].groupby(keys,as_index=False).value.median().rename(columns={"value":"hb_g_l"})
    derivations=[]
    for code,item,unit,minval,maxval,denominator in (("rbc_derived","*平均血红蛋白量","pg",10,60,"mch_pg"),
                                                   ("hct_cbc_derived","*平均血红蛋白浓度","g/l",150,500,"mchc_g_l")):
        component=cbc.loc[cbc["检验项名称"].eq(item)&cbc.unit_norm.eq(unit)&cbc.numeric.between(minval,maxval)]
        component=component.groupby(keys,as_index=False).numeric.median().rename(columns={"numeric":denominator})
        joint=hb.merge(component,on=keys,validate="one_to_one")
        joint["value"]=joint.hb_g_l/joint[denominator]*(100 if code=="hct_cbc_derived" else 1)
        joint=joint.loc[joint.value.between(*DEFINITIONS[code]["range"])]
        events[code]=collapse(joint);source_counts[code]=len(joint)
        derivations.append(joint.assign(indicator=code))
    direct=events["hct"];derived=events["hct_cbc_derived"]
    keys_direct=pd.MultiIndex.from_frame(direct[["hospital_id","timestamp_unix"]])
    supplemental=derived.loc[~pd.MultiIndex.from_frame(derived[["hospital_id","timestamp_unix"]]).isin(keys_direct)]
    events["hct_direct_or_cbc_derived"]=pd.concat([direct,supplemental],ignore_index=True);source_counts["hct_direct_or_cbc_derived"]=source_counts["hct"]+len(supplemental)
    cr=raw.loc[raw.base_valid&raw["标本名称"].isin(("血","血清","血浆"))&raw["检验项名称"].isin(("*肌酐(Cr)测定","*肌酐(Cr)测定-苦味酸法","*肌酐"))&raw.unit_norm.isin(("umol/l","mg/dl"))].copy()
    cr["cr_umol_l"]=cr.numeric*np.where(cr.unit_norm.eq("mg/dl"),88.4,1)
    cr["age"]=pd.to_numeric(cr["首页就诊时年龄"].str.extract(r"^\s*(\d+(?:\.\d+)?)\s*(?:岁|年)?\s*$")[0],errors="coerce")
    cr=cr.loc[cr.cr_umol_l.between(5,2000)&cr.age.between(18,120)&cr["首页性别"].isin(("男","女"))]
    ambiguity=cr.groupby(["hospital_id","timestamp_unix"])[["age","首页性别"]].nunique().gt(1).any(axis=1)
    cr=cr.loc[~pd.MultiIndex.from_frame(cr[["hospital_id","timestamp_unix"]]).isin(ambiguity.loc[ambiguity].index)]
    cr["value"]=egfr_2021(cr.cr_umol_l,cr.age,cr["首页性别"].eq("女"))
    events["egfr_cr_2021_derived"]=collapse(cr);source_counts["egfr_cr_2021_derived"]=len(cr)
    pd.concat(audits,ignore_index=True).to_csv(output/"source_field_audit.csv",index=False)
    pd.concat(derivations,ignore_index=True).to_csv(output/"cbc_derived_values.csv",index=False)
    catalog=raw.groupby(["检验项名称","单位"],dropna=False).size().reset_index(name="source_rows")
    catalog.to_csv(output/"all_item_unit_catalog.csv",index=False)
    return events,source_counts


def full_video_inventory():
    audit=pd.read_csv(SOURCE/"raw_video_audit.csv",dtype={"hospital_id":str,"video_id":str})
    valid=audit.loc[audit.status.isin(("retained_24h_pool","supported_lab_outside_24h","patient_without_supported_lab"))].copy()
    admissions=_load_hospital_episodes();rows=[]
    for row in valid.itertuples(index=False):
        matched=[ep for ep in admissions.get(row.hospital_id,()) if ep[0]<=row.capture_start_unix and row.capture_end_unix<=ep[1]]
        if len(matched)!=1:raise RuntimeError("Cached admission membership changed")
        mirror,local=row.video_id.split("_patient_")
        rows.append({"hospital_id":row.hospital_id,"video_id":row.video_id,"mirror":mirror,"lab_patient_id":int(local),
                     "capture_start_unix":row.capture_start_unix,"capture_end_unix":row.capture_end_unix,"admission_unix":matched[0][0],"discharge_unix":matched[0][1]})
    return pd.DataFrame(rows)


def match_events(events,videos):
    rows=[]
    for code,table in events.items():
        lookup={patient:list(zip(group.timestamp_unix,group.value)) for patient,group in table.groupby("hospital_id")}
        for video in videos.itertuples(index=False):
            eligible=[item for item in lookup.get(video.hospital_id,()) if video.admission_unix<=item[0]<=video.discharge_unix]
            selected=_nearest_measurement(eligible,video.capture_start_unix,video.capture_end_unix,24)
            if selected is None:continue
            rows.append({"indicator":code,**video._asdict(),"raw_value":selected["value"],"lab_time_unix":selected["timestamp_unix"],"match_delta_h":selected["delta_h"]})
    return pd.DataFrame(rows)


def frame_coverage(candidates):
    cached=set();failed=set()
    for directory in (STUDY/"common/cache/face224_20frame_main",STUDY/"exp9_face_interpolated_lab_regression/cache/combined_frames20",STUDY/"common/cache/face224_20frame"):
        path=directory/"frame_offsets.npz"
        if not path.exists():continue
        index=FrameOffsetIndex.load(path)
        if not _index_is_reusable(directory,index.video_ids,"20frame"):continue
        cached.update(index.video_ids)
        manifest=json.loads((directory/"index_manifest.json").read_text());failed.update(row["video_id"] for row in manifest["failed_videos"])
    pending=candidates.loc[~candidates.video_id.isin(cached|failed)].drop_duplicates("video_id")
    if len(pending):
        extra=build_or_reuse_frame_index(pending,str(HERE/"cache/new_indicator_frames20"),"20frame")
        cached.update(extra.video_ids)
    return cached


def main():
    output=HERE/"outputs";output.mkdir(parents=True,exist_ok=True)
    digest=sha256(LAB)
    protocol=json.loads((STUDY/"common/outputs/face_main_24h_frame_loss/protocol.json").read_text())
    if digest!=protocol["lab_table_sha256"]:raise RuntimeError("Raw video inventory is not current for this lab table")
    events,source_counts=load_events(output)
    videos=full_video_inventory();matched=match_events(events,videos)
    usable=frame_coverage(matched);matched["native224_20frame_eligible"]=matched.video_id.isin(usable)
    matched.to_csv(output/"matched_video_indicator_values.csv",index=False)
    summary=[]
    for code,definition in DEFINITIONS.items():
        labs=events.get(code,pd.DataFrame(columns=["hospital_id","timestamp_unix","value"]))
        candidate=matched.loc[matched.indicator.eq(code)];retained=candidate.loc[candidate.native224_20frame_eligible]
        summary.append({"indicator":code,"chinese_name":definition["cn"],"unit":definition["unit"],
                        "source_rows":source_counts.get(code,0),"valid_lab_events":len(labs),"lab_patients":labs.hospital_id.nunique(),
                        "matched24h_videos_before_frame_check":len(candidate),"exp2_videos":len(retained),"exp2_patients":retained.hospital_id.nunique(),
                        "matched_independent_lab_events":retained[["hospital_id","lab_time_unix"]].drop_duplicates().shape[0]})
    result=pd.DataFrame(summary);result.to_csv(output/"indicator_availability.csv",index=False)
    pd.concat([frame.assign(indicator=code) for code,frame in events.items()],ignore_index=True).to_csv(output/"canonical_indicator_events.csv",index=False)
    manifest={"lab_table_sha256":digest,"scope":"all currently validated same-admission Exp2 video sessions, not just the existing eight-task labelled subset",
              "matching":"one nearest report per video/analyte, <=24h to original capture interval; Session Timestamp basis; 20 valid native224 frames",
              "physiological_ranges":DEFINITIONS,"missing_units":"excluded; no implicit unit guessing","censored_values":"excluded from exact-value regression",
              "rbc_derivation":"Hb(g/L)/MCH(pg) in identical nonempty CBC suite, patient, report timestamp and specimen; rounded reconstruction, not a direct independent assay",
              "hct_derivation":"100*Hb(g/L)/MCHC(g/L) from identical CBC; direct reported HCT takes precedence",
              "hba1c_conversion":"NGSP(%)=0.09148*IFCC(mmol/mol)+2.152; direct percentage preferred for duplicate timestamps",
              "egfr_derivation":"adult CKD-EPI 2021: serum creatinine + same-record age/sex; conflicting demographic event metadata excluded; IDMS calibration not independently documented",
              "egfr_methods":"reported creatinine, reported cystatin-C and calculated 2021 values remain separate; not pooled",
              "no_model_training":True,"sources":["https://www.kidney.org/ckd-epi-creatinine-equation-2021","https://ngsp.org/ifcc.asp","https://media.beckmancoulter.com/-/media/diagnostics/products/hematology/dxh-500/docs/pdfs/dxh-500-series-casebook.pdf"]}
    if sha256(LAB)!=digest:raise RuntimeError("Lab table changed during the audit")
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(result.to_string(index=False),flush=True)


if __name__=="__main__":main()
