"""Prepare/check Exp9 by default; formal training requires explicit --train."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
import random
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex,_index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import RobustTargetScaler,fit_robust_target_scaler
from study.exp2_face_pretrained_head32_regression.train import train_task
from . import config
from .build_dataset import prepare,sha256
from .preoperative import nearest_preoperative


INDEX=DEVICE=None


def preflight():
    manifest=json.loads((config.OUTPUT/"experiment_manifest.json").read_text())
    if sha256(config.STUDY.parent/"merged_lab_tests.csv")!=manifest["lab_table_sha256"]:raise RuntimeError("Lab table changed")
    if manifest["training"]!=config.training_settings():raise RuntimeError("Exp2 training defaults changed since preparation")
    for target in config.TARGETS:
        if sha256(config.SOURCE/f"task_records/{target}.csv")!=manifest["source_task_record_sha256"][target]:
            raise RuntimeError("Exp2 reference patient assignments or labels changed")
    path=Path(manifest["frame_index_path"]);index=FrameOffsetIndex.load(path)
    if sha256(path)!=manifest["frame_index_sha256"] or not _index_is_reusable(path.parent,index.video_ids,"20frame"):
        raise RuntimeError("Shared native224 frame index changed")
    scalers=json.loads((config.OUTPUT/"target_scalers.json").read_text())["targets"]
    labs=pd.read_csv(config.SOURCE/"source_data/lab_timeseries.csv",dtype={"hospital_id":str},float_precision="round_trip")
    lookup={key:group.sort_values("timestamp_unix") for key,group in labs.groupby(["hospital_id","analyte"])}
    if sha256(config.OUTPUT/"target_scalers.json")!=manifest["scalers_sha256"]:raise RuntimeError("Scalers changed")
    for target in config.TARGETS:
        path=config.OUTPUT/f"task_records/{target}.csv"
        if sha256(path)!=manifest["task_records_sha256"][target]:raise RuntimeError("Interpolation records changed")
        records=pd.read_csv(path,dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
        if records.video_id.duplicated().any() or records.groupby("hospital_id").split.nunique().gt(1).any():raise RuntimeError("Video duplication or patient leakage")
        if not set(records.split)=={"train","val","test"}:raise RuntimeError("Missing split")
        pre=records.phase.eq("pre");post=records.phase.eq("post")
        if not (records.loc[post,"coverage_before_time_unix"]<records.loc[post,"capture_start_unix"]).all() or not (records.loc[post,"coverage_after_time_unix"]>records.loc[post,"capture_end_unix"]).all():
            raise RuntimeError("Interpolation lacks strict full-video bracketing")
        if not records.loc[post,"coverage_before_delta_h"].between(0,24).all() or not records.loc[post,"coverage_after_delta_h"].between(0,24).all():raise RuntimeError("Postoperative endpoints exceed 24h")
        if not ((pre & records.capture_end_unix.le(records.surgery_start_unix)) | (post & records.capture_start_unix.ge(records.surgery_end_unix))).all():
            raise RuntimeError("A video crosses the surgical boundary")
        if not (records.loc[pre,"support_right_time_unix"]<records.loc[pre,"surgery_start_unix"]).all():raise RuntimeError("Preoperative target uses surgical/postoperative labs")
        if not (records.loc[post,"support_left_time_unix"]>=records.loc[post,"surgery_end_unix"]).all():raise RuntimeError("Postoperative target uses pre/intraoperative labs")
        recomputed=records.loc[post,"support_left_value"]+(records.loc[post,"support_right_value"]-records.loc[post,"support_left_value"])*records.loc[post,"interpolation_alpha"]
        np.testing.assert_allclose(recomputed,records.loc[post,"raw_value"],rtol=1e-12,atol=1e-10)
        if not records.loc[pre,"selected_lab_time_unix"].ge(records.loc[pre,"admission_unix"]).all() or not records.loc[pre,"selected_lab_time_unix"].lt(records.loc[pre,"surgery_start_unix"]).all():
            raise RuntimeError("Preoperative target uses an intra/postoperative or other-admission report")
        prefix=config.reference.SCORE_DEFINITIONS[target]["value_column"].removesuffix("_value")
        for row in records.loc[pre].itertuples(index=False):
            events=lookup[(row.hospital_id,prefix)]
            label,status=nearest_preoperative(events.timestamp_unix.to_numpy(),events.value.to_numpy(),row.capture_start_unix,row.capture_end_unix,row.surgery_start_unix,row.admission_unix,row.discharge_unix)
            if status!="retained" or label["selected_lab_time_unix"]!=row.selected_lab_time_unix or label["raw_value"]!=row.raw_value:
                raise RuntimeError("Preoperative nearest report cannot be reproduced")
        fitted=fit_robust_target_scaler(target,records,config.reference.SCORE_DEFINITIONS[target]["unit"])
        if fitted.to_dict()!=scalers[target]:raise RuntimeError("Target scaling is not train-only")
        np.testing.assert_array_equal(fitted.transform(records.raw_value).astype(np.float32),records.robust_scaled_raw_value.to_numpy(np.float32))
        for video in records.video_id:
            left,right=index.frame_range(video)
            if right-left!=20:raise RuntimeError("Wrong selected-frame count")
        if manifest["split_policy"]=="reuse_exp2":
            existing=records.exp2_split.notna()
            if not records.loc[existing,"split"].eq(records.loc[existing,"exp2_split"]).all():raise RuntimeError("Exp2 patient assignment changed")
    print("[preflight-ok] observed preoperative nearest reports without time limit; postoperative interpolation unchanged; patient splits/scaling/native224 cache verified",flush=True)
    return manifest,index,scalers


def init_worker(queue,index_path):
    global INDEX,DEVICE
    DEVICE=int(queue.get());torch.cuda.set_device(DEVICE);torch.set_num_threads(1)
    INDEX=FrameOffsetIndex.load(index_path)


def job_seed(target):
    offset=int.from_bytes(hashlib.sha256(f"{config.reference.SEED}:efficientnet_b0:{target}".encode()).digest()[:4],"little")
    return (config.reference.SEED+offset)%(2**31-1)


def worker(job):
    from study.common.run_video_loss_12h import Tee
    from contextlib import redirect_stdout
    import sys
    target=job["target"];seed=job_seed(target)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    records=pd.read_csv(config.OUTPUT/f"task_records/{target}.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip")
    run=config.OUTPUT/f"runs/efficientnet_b0/{target}";run.mkdir(parents=True,exist_ok=True)
    with (run/"train.log").open("a",buffering=1) as log,redirect_stdout(Tee(sys.stdout,log)):
        train_task("efficientnet_b0",target,INDEX,records,RobustTargetScaler(**job["scaler"]),config.reference.WEIGHTS_DIR,str(run),
                   train_batch_policy=config.BATCH_POLICY,loss_level=config.LOSS_LEVEL)
    checkpoint=torch.load(run/"model.pt",map_location="cpu",weights_only=True)
    if checkpoint["loss_level"]!="frame" or checkpoint["train_batch_policy"]!=config.BATCH_POLICY:raise RuntimeError("Wrong saved training protocol")
    checkpoint["label_source"]="preoperative nearest observed / postoperative phase-separated linear interpolation"
    checkpoint["prepared_contract_sha256"]=job["contract"]
    torch.save(checkpoint,run/"model.pt")
    predictions=pd.read_csv(run/"video_predictions.csv",dtype={"hospital_id":str,"video_id":str},float_precision="round_trip").sort_values("video_id").reset_index(drop=True)
    expected=records.sort_values("video_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(predictions[["hospital_id","video_id","split"]],expected[["hospital_id","video_id","split"]],check_dtype=False)
    np.testing.assert_allclose(predictions.y_true,expected.raw_value,rtol=1e-12,atol=1e-10)
    if not predictions.frame_count.eq(20).all():raise RuntimeError("Evaluation dropped selected frames")
    (run/"job_complete.json").write_text(json.dumps({"contract":job["contract"],"seed":seed})+"\n")
    torch._dynamo.reset();torch.cuda.empty_cache()
    return {"target":target,"architecture":"efficientnet_b0","seed":seed,"status":"ok"}


def train(manifest,scalers):
    contract=sha256(config.OUTPUT/"experiment_manifest.json")
    jobs=[];rows=[]
    for target in config.TARGETS:
        run=config.OUTPUT/f"runs/efficientnet_b0/{target}";marker=run/"job_complete.json"
        expected={"contract":contract,"seed":job_seed(target)}
        if marker.exists() and json.loads(marker.read_text())==expected and all((run/name).is_file() for name in ("model.pt","history.csv","metrics.csv","video_predictions.csv")):
            rows.append({"target":target,"architecture":"efficientnet_b0","seed":job_seed(target),"status":"ok"})
        else:jobs.append({"target":target,"scaler":scalers[target],"contract":contract})
    count=min(4,torch.cuda.device_count(),len(jobs))
    if jobs and not count:raise RuntimeError("Formal Exp9 training requires CUDA")
    print(f"[scheduler] interpolation regression jobs={len(jobs)} GPUs={count}",flush=True)
    (config.OUTPUT/"COMPLETE").unlink(missing_ok=True)
    (config.OUTPUT/"run_manifest.json").write_text(json.dumps({"training_started":True,"prepared_contract_sha256":contract,"job_seeds":{t:job_seed(t) for t in config.TARGETS}},indent=2)+"\n")
    ctx=mp.get_context("spawn")
    if jobs:
        with ctx.Manager() as manager:
            queue=manager.Queue()
            for gpu in range(count):queue.put(gpu)
            with ProcessPoolExecutor(max_workers=count,mp_context=ctx,initializer=init_worker,initargs=(queue,manifest["frame_index_path"])) as pool:
                futures={pool.submit(worker,job):job for job in jobs}
                for future in as_completed(futures):
                    job=futures[future]
                    try:row=future.result()
                    except Exception:
                        row={"target":job["target"],"status":"failed","error":traceback.format_exc()};print(row["error"],flush=True)
                    rows.append(row);pd.DataFrame(rows).to_csv(config.OUTPUT/"run_index.csv",index=False)
                    print(f"[task-finished] {job['target']} {row['status']}",flush=True)
    if any(row["status"]!="ok" for row in rows):raise RuntimeError("Exp9 jobs failed; see run_index.csv")
    for name in ("history","metrics"):
        pd.concat([pd.read_csv(config.OUTPUT/f"runs/efficientnet_b0/{target}/{name}.csv") for target in config.TARGETS],ignore_index=True).to_csv(config.OUTPUT/f"{name}_all.csv",index=False)
    from study.exp2_face_pretrained_head32_regression.plot_results import main as plot_results
    plot_results(config.OUTPUT)
    from .plots import plot_main_comparison
    plot_main_comparison(config.OUTPUT,config.SOURCE)
    (config.OUTPUT/"COMPLETE").write_text("interpolation models, histories, metrics and figures completed\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",action="store_true",help="Explicitly start formal model fitting")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--split-policy",choices=("reuse_exp2","balanced_search"),default=config.DEFAULT_SPLIT_POLICY)
    args=parser.parse_args()
    config.OUTPUT.mkdir(parents=True,exist_ok=True)
    with (config.OUTPUT/".experiment.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.train and not args.check_only:
            manifest=prepare(args.split_policy)
            from .plots import plot_preparation
            plot_preparation()
        manifest,index,scalers=preflight()
        if args.train:train(manifest,scalers)
        else:print("[prepared-only] no training or GPU jobs started",flush=True)


if __name__=="__main__":main()
