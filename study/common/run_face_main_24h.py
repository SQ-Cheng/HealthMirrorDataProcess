"""Matched face-only main experiments: 24h, frame loss, 12 lab events/batch."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from . import run_video_loss_12h as worker_api
from .video_loss import DistinctLabViewBatchSampler
from study.exp2_face_pretrained_head32_regression import config
from study.exp2_face_pretrained_head32_regression.data import validate_source_data
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex, _index_is_reusable
from study.exp2_face_pretrained_head32_regression.scaling import fit_robust_target_scaler, RobustTargetScaler


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
SOURCE = Path(config.OUTPUT_DIRS["20frame"])
INDEX_DIR = STUDY / "common/cache/face224_20frame_main"
INDEX_PATH = INDEX_DIR / "frame_offsets.npz"
STATE = STUDY / "common/outputs/face_main_24h_frame_loss"
OUTPUTS = {"regression": SOURCE, "classification": STUDY / "exp2_face_pretrained_head32_classification/outputs/face224"}


def prepare(families, overwrite):
    if "regression" in families:
        command = [sys.executable,"-u","-m","study.exp2_face_pretrained_head32_regression.run_all",
                   "--prepare-only","--frame-policy","20frame","--match-max-delta-hours","24",
                   "--loss-level","frame","--train-batch-policy","distinct_lab_views",
                   "--index-dir",str(INDEX_DIR),"--reference-output-dir",""]
        if overwrite:command.append("--overwrite")
        print("[prepare] current full lab table; fresh shared 24h cohort and split search",flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    if "classification" in families:
        destination=OUTPUTS["classification"]
        if (destination/"metrics_all.csv").exists() and not overwrite:
            raise FileExistsError("Classification results exist; use --overwrite")
        if overwrite and destination.exists():shutil.rmtree(destination)
        (destination/"task_records").mkdir(parents=True,exist_ok=True)
        for target in config.TARGETS:
            shutil.copy2(SOURCE/f"task_records/{target}.csv",destination/f"task_records/{target}.csv")
        for name in ("target_scalers.json","task_summary.csv","split_assignment_manifest.json","split_distribution_audit.csv","split_distribution_pairwise.csv"):
            shutil.copy2(SOURCE/name,destination/name)
    manifest=json.loads((SOURCE/"experiment_manifest.json").read_text())
    lab=ROOT/"merged_lab_tests.csv"
    fingerprint=manifest["source_data_quality_report"]["source_fingerprints"]["merged_lab_tests"]
    if fingerprint["size_bytes"]!=lab.stat().st_size or fingerprint["mtime_ns"]!=lab.stat().st_mtime_ns:
        raise RuntimeError("Main regression data do not use the current lab table")
    protocol={
        "lab_table_sha256":worker_api.sha256(lab),"matching_hours":24,"loss_level":"frame",
        "train_batch_policy":"distinct_lab_views","frame_index":str(INDEX_PATH),
        "frame_index_sha256":worker_api.sha256(INDEX_PATH),"targets":list(config.TARGETS),
        "frame_batch_size":240,"batch_unique_measurements":12,"frames_per_video":20,
        "views_per_video_per_batch":1,"training_views":list(config.VIEW_NAMES),
        "losses_per_full_batch":240,"epoch_coverage":"all twenty frames and five views per video, no dropping/oversampling",
        "clinical_event_identity":"hospital_id + target-specific matched report timestamp",
        "split_policy":"512-candidate patient-disjoint distribution search; exact per-target split shared between families",
        "head_lr":config.HEAD_LEARNING_RATE,"head_epochs":config.HEAD_MAX_EPOCHS,"head_patience":config.HEAD_PATIENCE,
        "finetune_lr":config.FINETUNE_LEARNING_RATE,"finetune_epochs":config.FINETUNE_MAX_EPOCHS,"finetune_patience":config.FINETUNE_PATIENCE,
        "min_lr":config.MIN_LEARNING_RATE,"weight_decay":config.WEIGHT_DECAY,"optimizer":"AdamW",
        "scheduler":"cosine, no warmup","compile_mode":config.TORCH_COMPILE_MODE,
        "train_loss":"per-frame weighted BCEWithLogitsLoss / unweighted SmoothL1(beta=0.5)",
        "validation_loss":"per-frame; original view only",
        "reported_metrics":"video-level mean of twenty original frame probabilities/predictions",
        "source_records_sha256":{target:worker_api.sha256(SOURCE/f"task_records/{target}.csv") for target in config.TARGETS},
        "scalers_sha256":worker_api.sha256(SOURCE/"target_scalers.json"),
    }
    STATE.mkdir(parents=True,exist_ok=True)
    (STATE/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    (STATE/"COMPLETE").unlink(missing_ok=True)
    for family in families:
        (OUTPUTS[family]/"main_protocol.json").write_text(json.dumps({**protocol,"family":family},indent=2)+"\n")
        (OUTPUTS[family]/"COMPLETE").unlink(missing_ok=True)
    if "classification" in families:
        classifier={**protocol,"family":"classification","architecture":"efficientnet_b0","head_hidden_features":32,
                    "reference_source":str(SOURCE),"pos_weight":"training negative/positive video counts; identical ratio for equal frame/view coverage",
                    "clinical_thresholds":config.SCORE_DEFINITIONS,"source_data_quality_report":manifest["source_data_quality_report"]}
        (OUTPUTS["classification"]/"experiment_manifest.json").write_text(json.dumps(classifier,indent=2)+"\n")
    (STATE/"DATA_READY").write_text("fresh matched main cohorts and protocols\n")


def preflight(families):
    protocol=json.loads((STATE/"protocol.json").read_text())
    if protocol["lab_table_sha256"]!=worker_api.sha256(ROOT/"merged_lab_tests.csv"):
        raise RuntimeError("Lab table changed after preparation")
    validate_source_data(SOURCE/"source_data",24)
    index=FrameOffsetIndex.load(INDEX_PATH)
    if set(index.video_formats)!={"ffv1"} or not _index_is_reusable(INDEX_DIR,index.video_ids,"20frame"):
        raise RuntimeError("Native224 frame index is stale")
    if worker_api.sha256(INDEX_PATH)!=protocol["frame_index_sha256"]:
        raise RuntimeError("Frame index changed")
    scalers=json.loads((SOURCE/"target_scalers.json").read_text())["targets"]
    audits=[]
    for target in config.TARGETS:
        path=SOURCE/f"task_records/{target}.csv"
        if worker_api.sha256(path)!=protocol["source_records_sha256"][target]:raise RuntimeError(f"Records changed: {target}")
        records=pd.read_csv(path,dtype={"hospital_id":str,"video_id":str})
        assert not records.video_id.duplicated().any()
        assert not records.groupby("hospital_id").split.nunique().gt(1).any()
        assert records.match_delta_h.between(0,24+1e-9).all()
        assert all(index.frame_range(video)[1]-index.frame_range(video)[0]==20 for video in records.video_id)
        fitted=fit_robust_target_scaler(target,records,config.SCORE_DEFINITIONS[target]["unit"])
        if fitted.to_dict()!=scalers[target]:raise RuntimeError("Scaler is not train-only")
        np.testing.assert_array_equal(fitted.transform(records.raw_value).astype(np.float32),records.robust_scaled_raw_value.to_numpy(np.float32))
        for family in families:
            if worker_api.sha256(OUTPUTS[family]/f"task_records/{target}.csv")!=protocol["source_records_sha256"][target]:
                raise RuntimeError(f"Family records/splits differ: {family}/{target}")
        for split,group in records.groupby("split"):
            assert group.binary_label.nunique()==2
            audits.append({"target":target,"split":split,"videos":len(group),"patients":group.hospital_id.nunique(),"lab_events":group.clinical_event_id.nunique()})
        group=records.loc[records.split.eq("train")].reset_index(drop=True)
        dataset=SimpleNamespace(expand_all_views=False,video_records=group,frame_video_rows=np.repeat(np.arange(len(group)),20),views=config.VIEW_NAMES)
        batches=list(DistinctLabViewBatchSampler(dataset,240))
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)),np.arange(len(group)*100))
        for batch in batches:
            groups=np.asarray(batch).reshape(-1,20)
            rows=groups[:,0]//100
            assert group.iloc[rows].clinical_event_id.nunique()==len(rows)
            for selected in groups:
                assert len(set(selected//100))==1 and len(set(selected%5))==1
        assert all(len(batch)==240 for batch in batches[:-1])
    pd.DataFrame(audits).to_csv(STATE/"cohort_counts.csv",index=False)
    print("[preflight-ok] 24h, shared patient split, train-only scaling, 12 distinct labs / 240 frame losses; all frames/views retained",flush=True)
    return index,scalers,protocol


def init_worker(queue):
    worker_api.GPU=int(queue.get())
    torch.cuda.set_device(worker_api.GPU)
    worker_api.INDEX=FrameOffsetIndex.load(INDEX_PATH)


def finalize(family):
    root=OUTPUTS[family]
    for name in ("metrics","history"):
        pd.concat([pd.read_csv(root/f"runs/efficientnet_b0/{target}/{name}.csv") for target in config.TARGETS],ignore_index=True).to_csv(root/f"{name}_all.csv",index=False)
    if family=="regression":
        from study.exp2_face_pretrained_head32_regression.plot_results import main as plot
        plot(root)
    else:
        from .selected_5fold_plots import plot_fold_classification
        plot_fold_classification(root,title="Face-only classification | 24h | held-out test set")
        import matplotlib.pyplot as plt
        from .plot_layout import target_grid_shape,target_grid_figsize
        from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS
        rows,cols=target_grid_shape(len(config.TARGETS))
        figure,axes=plt.subplots(rows,cols,figsize=target_grid_figsize(rows,cols),squeeze=False)
        history=pd.read_csv(root/"history_all.csv")
        for axis,target in zip(axes.flat,config.TARGETS):
            values=history.loc[history.target.eq(target)]
            axis.plot(values.global_epoch,values.train_loss,label="Train",color="#2878B5")
            axis.plot(values.global_epoch,values.val_loss,label="Validation",color="#CB6547")
            axis.set(title=TASK_LABELS[target],xlabel="Epoch",ylabel="Frame-level weighted BCE")
            axis.legend(fontsize=8);axis.grid(alpha=.2)
        figure.tight_layout();figure.savefig(root/"figures/training_history.png",dpi=180);plt.close(figure)
    (root/"COMPLETE").write_text("eight main models and figures completed\n")


def smoke(index,scalers):
    from study.exp2_binary_classification_common.engine import train_task as classify
    from study.exp2_face_pretrained_head32_regression import train as regress
    target="lactate_high"
    records=pd.read_csv(SOURCE/f"task_records/{target}.csv",dtype={"hospital_id":str,"video_id":str})
    train=records.loc[records.split.eq("train")].drop_duplicates("clinical_event_id").groupby("binary_label",group_keys=False).head(6)
    assert len(train)==12
    held=records.loc[records.split.ne("train")].groupby(["split","binary_label"],group_keys=False).head(1)
    subset=pd.concat([train,held],ignore_index=True)
    with tempfile.TemporaryDirectory(prefix="main24h_frame_loss_smoke_") as temporary:
        root=Path(temporary);path=root/"records.csv";subset.to_csv(path,index=False)
        classify("face_only",target,0,config.SEED,smoke=True,output_dir=root/"classification",records_path=path,
                 reference_records_path=SOURCE/f"task_records/{target}.csv",frame_index_path=INDEX_PATH,
                 train_batch_policy="distinct_lab_views",loss_level="frame")
        with patch.object(regress,"TORCH_COMPILE_ENABLED",False):
            regress.train_task("efficientnet_b0",target,index,subset,RobustTargetScaler(**scalers[target]),config.WEIGHTS_DIR,str(root/"regression"),
                               head_epochs=1,finetune_epochs=1,max_batches=1,train_batch_policy="distinct_lab_views",loss_level="frame")
        for family in ("classification","regression"):
            run=root/(f"classification/runs/efficientnet_b0/{target}" if family=="classification" else "regression")
            saved=torch.load(run/"model.pt",map_location="cpu",weights_only=True)
            assert saved["loss_level"]=="frame" and saved["train_batch_policy"]=="distinct_lab_views"
    print("[smoke-ok] both frame objectives, 12-event batches, two stages, native224 frames and saved checkpoints",flush=True)


def main(default_families=("classification","regression")):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families",default=",".join(default_families))
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--reuse-prepared",action="store_true")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    families=tuple(args.families.split(","))
    if not families or set(families)-set(OUTPUTS):parser.error("families must be classification and/or regression")
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/".queue.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.reuse_prepared:prepare(families,args.overwrite)
        index,scalers,protocol=preflight(families)
        if args.prepare_only:return
        if args.smoke:smoke(index,scalers);return
        contract=worker_api.sha256(STATE/"protocol.json")
        jobs=[{"family":family,"target":target,"output":str(OUTPUTS[family]),"batch_policy":"distinct_lab_views",
               "loss_level":"frame","reference_source":str(SOURCE),"frame_index_path":str(INDEX_PATH),"scaler":scalers[target],"contract":contract}
              for target in config.TARGETS for family in families]
        workers=min(4,torch.cuda.device_count(),len(jobs))
        if not workers:raise RuntimeError("CUDA required for main training")
        print(f"[scheduler] 24h frame-loss main jobs={len(jobs)} GPUs={workers}",flush=True)
        rows=[];ctx=mp.get_context("spawn")
        with ctx.Manager() as manager:
            queue=manager.Queue()
            for gpu in range(workers):queue.put(gpu)
            with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=init_worker,initargs=(queue,)) as pool:
                futures={pool.submit(worker_api.train_one,job):job for job in jobs}
                for future in as_completed(futures):
                    job=futures[future]
                    try:row=future.result()
                    except Exception:
                        row={"family":job["family"],"target":job["target"],"status":"failed","error":traceback.format_exc()}
                        print(row["error"],flush=True)
                    rows.append(row)
                    pd.DataFrame([r for r in rows if r["family"]==job["family"]]).to_csv(OUTPUTS[job["family"]]/"run_index.csv",index=False)
                    print(f"[task-finished] {job['family']}/{job['target']} {row['status']}",flush=True)
        if any(row["status"]!="ok" for row in rows):raise RuntimeError("Main jobs failed; see run_index.csv")
        for family in families:finalize(family)
        preflight(families)
        (STATE/"COMPLETE").write_text("matched 24h frame-loss main models and figures completed\n")


if __name__=="__main__":main()
