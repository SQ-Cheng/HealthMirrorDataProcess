"""Frozen EN-B0 head32/head64 controls matched to Exp2 and Exp6 DINO runs."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import fcntl
import gc
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from study.common.run_video_loss_12h import Tee, sha256
from study.common.video_loss import DistinctLabViewBatchSampler, VideoViewBatchSampler
from study.exp2_face_dinov3_frozen import run_main_regression as exp2
from study.exp2_face_dinov3_frozen.data import DinoFeatureDataset, no_feature_fitting
from study.exp2_face_dinov3_frozen.features import ensure_cache, load_frozen_encoder
from study.exp2_face_architecture_ablation.train import train_task as train_single
from study.exp2_face_pretrained_head32_regression.models import SingleTaskHead
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp6_face_pair_dinov3_regression import run as exp6
from study.exp6_face_pair_dinov3_regression.data import FeaturePairs
from study.exp6_face_pair_dinov3_regression.train import train_task as train_pairs


STUDY = Path(__file__).resolve().parents[1]
STATE = STUDY / "common/outputs/frozen_en_regression_controls"
CACHE = STUDY / "common/cache/efficientnet_b0_frozen_exp2_exp6"
ARCHITECTURE = "efficientnet_b0_frozen"
WIDTHS = (32, 64)
ROOTS = {"exp2": exp2.config.HERE / "outputs/frozen_en_b0", "exp6": exp6.config.OUTPUT / "frozen_en_b0"}
INDEX2 = INDEX6 = MAPPING = DEVICE = None


def build_head(hidden):
    return SingleTaskHead(1280, hidden, dropout=.25)


def head_parameters(hidden):
    return 1284 * hidden + 1


def result_root(family, hidden):
    root = ROOTS[family] / f"head{hidden}"
    return root / "regression" / ARCHITECTURE if family == "exp2" else root


def frame_mapping(source, cache):
    mapping = np.full(len(source.starts), -1, dtype=np.int64)
    for video in source.video_ids:
        a, b = source.video_lookup[str(video)], cache.video_lookup[str(video)]
        for key in ("video_paths", "video_formats", "codec_extradata"):
            if getattr(source, key)[a] != getattr(cache, key)[b]:
                raise RuntimeError(f"Video source differs between experiments: {video}")
        start, end = source.frame_range(video)
        other_start, other_end = cache.frame_range(video)
        for key in ("starts", "ends", "source_indices"):
            np.testing.assert_array_equal(getattr(source, key)[start:end], getattr(cache, key)[other_start:other_end])
        mapping[start:end] = np.arange(other_start, other_end)
    if (mapping < 0).any():
        raise RuntimeError("Some Exp2 frames have no exact shared cache mapping")
    return mapping


def single_loader(index, records, architecture, family, train):
    dataset = DinoFeatureDataset(index, records, exp2.config.VIEWS if train else ("original",), family, CACHE)
    dataset.frame_indices = MAPPING[dataset.frame_indices]
    if dataset.features.shape != (len(INDEX6.starts), 5, 1280):
        raise RuntimeError("Frozen EN-B0 features have the wrong shape")
    sampler = DistinctLabViewBatchSampler(dataset, 240) if train else VideoViewBatchSampler(dataset, 500, False)
    return dataset, DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())


def pair_loader(index, records, train, cache_dir=CACHE):
    dataset = FeaturePairs(index, records, exp6.config.VIEWS if train else ("original",), cache_dir, feature_count=1280)
    if train:
        return dataset, DataLoader(dataset, batch_sampler=DistinctLabViewBatchSampler(dataset, 240),
                                   num_workers=0, pin_memory=torch.cuda.is_available())
    return dataset, DataLoader(dataset, batch_size=480, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def single_config(output, epochs=None):
    options = vars(exp2.training_config(output, epochs)).copy()
    options.update(LEARNING_RATES={ARCHITECTURE: 2e-4}, MIN_LEARNING_RATES={ARCHITECTURE: 1e-6})
    return SimpleNamespace(**options)


def audit():
    exp2.configure_variant(32)
    index2, scalers2, hashes2, counts2, lab_hash = exp2.preflight()
    records6, scalers6, index6, main6, counts6 = exp6.preflight()
    frame_mapping(index2, index6)
    manifests = {}
    for family in ("exp2", "exp6"):
        if family == "exp2":
            references = {width: exp2.config.HERE / ("outputs/main24h_frame_loss" + ("_head64" if width == 64 else "")) for width in WIDTHS}
            hashes, scalers, index_path = hashes2, scalers2, exp2.INDEX_PATH
        else:
            references = {width: exp6.config.OUTPUT for width in WIDTHS}
            hashes, scalers, index_path = main6["records_sha256"], scalers6, Path(main6["frame_index"])
        for path in set(references.values()):
            if not (path / "COMPLETE").is_file():
                raise RuntimeError(f"DINO comparator is incomplete: {path}")
        dino = json.loads((references[32] / "experiment_manifest.json").read_text())
        expected = {"learning_rate": 2e-4, "minimum_lr": 1e-6, "max_epochs": 80,
                    "patience": 12, "weight_decay": 1e-3, "dropout": .25, "stages": 1, "seed": exp6.config.SEED}
        for key, value in expected.items():
            if key == "seed" and family == "exp2":
                continue
            if dino.get(key) != value:
                raise RuntimeError(f"DINO training setting differs: {family}/{key}")
        source_key = "source_records_sha256" if family == "exp2" else "records_sha256"
        if dino[source_key] != hashes or dino["lab_table_sha256"] != lab_hash:
            raise RuntimeError("DINO clinical data differ from the current baseline")
        if family == "exp2":
            wide = json.loads((references[64] / "experiment_manifest.json").read_text())
            if any(wide[key] != dino[key] for key in (*expected.keys(), source_key, "job_seeds") if key != "seed"):
                raise RuntimeError("DINO head32/head64 protocols differ")
        manifests[family] = {
            "family": family, "architecture": ARCHITECTURE, "backbone_frozen": True,
            "backbone_weight_sha256": sha256(exp2.config.WEIGHTS.parent / "efficientnet_b0_rwightman-7f5810bc.pth"),
            "backbone_parameters": 4007548, "feature": "1280-dimensional global-average-pool; eval-mode frozen BatchNorm",
            "head_widths": list(WIDTHS), "head_parameters": {str(width): head_parameters(width) for width in WIDTHS},
            "matching_hours": 24, "targets": list(hashes), "source_records_sha256": hashes,
            "lab_table_sha256": lab_hash, "scalers": scalers, "frame_index": str(index_path),
            "frame_index_sha256": sha256(index_path), "feature_cache_index": str(main6["frame_index"]),
            "feature_cache_index_sha256": main6["frame_index_sha256"], "views": list(exp2.config.VIEWS),
            "frame_inputs_per_batch": 240, "distinct_measurements_or_pairs_per_batch": 12, "frames_per_video": 20,
            "loss": "unweighted per-frame SmoothL1(beta=0.5)" if family == "exp2" else "patient-weighted per-frame-pair SmoothL1(beta=0.5); two 120-pair microbatches per logical batch",
            "evaluation": "mean predictions over twenty original frame inputs; authoritative raw label/delta",
            "fusion": "single-face feature" if family == "exp2" else "late GAP minus early GAP",
            "learning_rate": 2e-4, "minimum_lr": 1e-6, "max_epochs": 80, "patience": 12,
            "weight_decay": 1e-3, "dropout": .25, "gradient_clip": 1., "optimizer": "AdamW",
            "scheduler": "cosine; no warmup", "stages": 1, "compile": False,
            "job_seeds": {target: exp2.reference.job_seed("regression", target) if family == "exp2" else exp6.config.SEED for target in hashes},
            "dino_manifests_sha256": {str(width): sha256(path / "experiment_manifest.json") for width, path in references.items()},
        }
    return (index2, index6), manifests, {"exp2": counts2, "exp6": counts6}


def init_worker(queue, index2_path, index6_path):
    global INDEX2, INDEX6, MAPPING, DEVICE
    INDEX2, INDEX6 = FrameOffsetIndex.load(index2_path), FrameOffsetIndex.load(index6_path)
    MAPPING = frame_mapping(INDEX2, INDEX6)
    gpu = int(queue.get())
    torch.cuda.set_device(gpu)
    DEVICE = torch.device(f"cuda:{gpu}")


def train_job(job, records=None, output=None, epochs=None):
    family, width, target = job["family"], job["width"], job["target"]
    root = ROOTS[family] / f"head{width}" if output is None else Path(output)
    if family == "exp2":
        if records is not None:
            (root / "source_records").mkdir(parents=True, exist_ok=True)
            records.to_csv(root / f"source_records/{target}.csv", index=False)
        train_single({"architecture": ARCHITECTURE, "family": "regression", "target": target, "scaler": job["scaler"]},
                     INDEX2, DEVICE, experiment_config=single_config(root, epochs),
                     model_factory=lambda _: build_head(width), loader_factory=single_loader, feature_scaler=no_feature_fitting)
        run = root / "regression" / ARCHITECTURE / "runs" / target
    else:
        if records is None:
            records = pd.read_csv(ROOTS[family] / f"task_records/{target}.csv", dtype={"hospital_id": str}, float_precision="round_trip")
        run = root / "runs" / target
        train_pairs(target, width, records, job["scaler"], INDEX6, DEVICE, run,
                    {"feature": "1280-dimensional GAP", "fusion": "late GAP minus early GAP"}, epochs=epochs, cache_dir=CACHE,
                    model_factory=build_head, loader_factory=pair_loader, expected_parameters=head_parameters(width))
    return run


def worker(job):
    run = result_root(job["family"], job["width"]) / "runs" / job["target"]
    run.mkdir(parents=True, exist_ok=True)
    with (run / "train.log").open("a", buffering=1) as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        train_job(job)
    saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    parameter_key = "parameters" if job["family"] == "exp2" else "head_parameters"
    if saved[parameter_key] != head_parameters(job["width"]):
        raise RuntimeError("Frozen EN head parameter count mismatch")
    saved.update(backbone_frozen=True, head_hidden=job["width"], feature_count=1280,
                 backbone_weight_sha256=job["weight_sha256"], experiment_contract=job["contract"],
                 source_records_sha256=job["records_sha256"], feature_cache=str(CACHE))
    torch.save(saved, run / "model.pt")
    (run / "job_complete.json").write_text(json.dumps({"contract": job["contract"]}) + "\n")
    gc.collect()
    torch.cuda.empty_cache()
    return {"family": job["family"], "head_hidden": job["width"], "target": job["target"], "status": "ok"}


def smoke(manifests):
    global DEVICE
    DEVICE = torch.device("cuda:0")
    for family, target in (("exp2", "lactate_high"), ("exp6", "hemoglobin_low")):
        path = (exp2.BASELINE if family == "exp2" else exp6.config.BASELINE) / f"task_records/{target}.csv"
        table = pd.read_csv(path, dtype={"hospital_id": str}, float_precision="round_trip" if family == "exp6" else None)
        training = table.loc[table.split.eq("train")]
        training = (training.drop_duplicates("clinical_event_id").groupby("binary_label", group_keys=False).head(6)
                    if family == "exp2" else training.head(12))
        subset = pd.concat([training, table.loc[table.split.ne("train")].groupby("split", group_keys=False).head(2)], ignore_index=True)
        with tempfile.TemporaryDirectory(prefix=f"frozen_en_{family}_smoke_") as name:
            for width in WIDTHS:
                run = train_job({"family": family, "width": width, "target": target,
                                 "scaler": manifests[family]["scalers"][target]}, subset, Path(name) / f"head{width}", epochs=1)
                saved = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
                assert saved["parameters" if family == "exp2" else "head_parameters"] == head_parameters(width)
                assert pd.read_csv(run / "history.csv").train_model_inputs.iloc[0] == 1200
    print("[smoke-ok] both experiments and head widths; exact frozen features, batch coverage, losses and checkpoints", flush=True)


def main():
    global INDEX2, INDEX6, MAPPING
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    (INDEX2, INDEX6), manifests, counts = audit()
    MAPPING = frame_mapping(INDEX2, INDEX6)
    if args.check_only:
        for family, count in counts.items():
            print(family, count.to_string(index=False))
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.plot_only:
            ensure_cache(Path(manifests["exp6"]["frame_index"]), CACHE, architecture="efficientnet_b0")
            if args.smoke:
                smoke(manifests)
                return
            jobs, rows = [], []
            for family, manifest in manifests.items():
                root = ROOTS[family]
                root.mkdir(parents=True, exist_ok=True)
                path = root / "experiment_manifest.json"
                if path.exists() and json.loads(path.read_text()) != manifest:
                    raise RuntimeError(f"Existing frozen EN contract differs: {family}")
                path.write_text(json.dumps(manifest, indent=2) + "\n")
                counts[family].to_csv(root / "cohort_counts.csv", index=False)
                (root / "task_records").mkdir(exist_ok=True)
                source = exp2.BASELINE if family == "exp2" else exp6.config.BASELINE
                for target in manifest["targets"]:
                    shutil.copy2(source / f"task_records/{target}.csv", root / f"task_records/{target}.csv")
                for width in WIDTHS:
                    if family == "exp2":
                        (root / f"head{width}/source_records").mkdir(parents=True, exist_ok=True)
                        for target in manifest["targets"]:
                            shutil.copy2(root / f"task_records/{target}.csv", root / f"head{width}/source_records/{target}.csv")
                    contract = sha256(path)
                    for target in manifest["targets"]:
                        run = result_root(family, width) / "runs" / target
                        marker = run / "job_complete.json"
                        if (marker.exists() and json.loads(marker.read_text()) == {"contract": contract}
                                and all((run / file).is_file() for file in ("model.pt", "history.csv", "metrics.csv",
                                          "video_predictions.csv" if family == "exp2" else "pair_predictions.csv"))):
                            rows.append({"family": family, "head_hidden": width, "target": target, "status": "ok"})
                        else:
                            jobs.append({"family": family, "width": width, "target": target, "scaler": manifest["scalers"][target],
                                         "contract": contract, "weight_sha256": manifest["backbone_weight_sha256"],
                                         "records_sha256": manifest["source_records_sha256"][target]})
            workers = min(4, torch.cuda.device_count(), len(jobs))
            if jobs and not workers:
                raise RuntimeError("CUDA required")
            print(f"[scheduler] frozen EN-B0 jobs={len(jobs)} GPUs={workers} Exp2=20 Exp6=22", flush=True)
            ctx = mp.get_context("spawn")
            if jobs:
                with ctx.Manager() as manager:
                    queue = manager.Queue()
                    for gpu in range(workers):
                        queue.put(gpu)
                    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                             initargs=(queue, manifests["exp2"]["frame_index"], manifests["exp6"]["frame_index"])) as pool:
                        futures = {pool.submit(worker, job): job for job in jobs}
                        for future in as_completed(futures):
                            job = futures[future]
                            try:
                                row = future.result()
                            except Exception:
                                row = {"family": job["family"], "head_hidden": job["width"], "target": job["target"],
                                       "status": "failed", "error": traceback.format_exc()}
                                print(row["error"], flush=True)
                            rows.append(row)
                            pd.DataFrame(rows).to_csv(STATE / "run_index.csv", index=False)
                            print(f"[task-finished] {job['family']}/head{job['width']}/{job['target']} {row['status']}", flush=True)
            pd.DataFrame(rows).to_csv(STATE / "run_index.csv", index=False)
            if any(row["status"] != "ok" for row in rows):
                raise RuntimeError("Frozen EN jobs failed; see run_index.csv")
            for family, manifest in manifests.items():
                for width in WIDTHS:
                    root = result_root(family, width)
                    for name in ("history", "metrics"):
                        pd.concat([pd.read_csv(root / f"runs/{target}/{name}.csv") for target in manifest["targets"]], ignore_index=True).to_csv(root / f"{name}_all.csv", index=False)
        if audit()[1] != manifests:
            raise RuntimeError("Clinical or DINO reference changed during EN training")
        from .plot_frozen_en_regression_controls import plot_all
        plot_all(manifests)
        for root in ROOTS.values():
            (root / "COMPLETE").write_text("frozen EN head32/head64 controls and five-model plots completed\n")
        (STATE / "COMPLETE").write_text("42 frozen EN heads and both five-model comparisons completed\n")
        print("[queue-complete] frozen EN Exp2/Exp6 regression controls and five-model figures", flush=True)


if __name__ == "__main__":
    main()
