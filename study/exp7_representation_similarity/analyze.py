"""Test whether held-out lab proximity tracks learned face-feature proximity."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from torch.utils.data import DataLoader

from study.exp2_face_pretrained_head32_regression import config as face_config
from study.exp2_face_pretrained_head32_regression.data import AllFramesDataset
from study.exp2_face_pretrained_head32_regression.frame_index import FrameOffsetIndex
from study.exp2_face_pretrained_head32_regression.models import build_pretrained_model as build_face
from study.exp2_face_pretrained_head32_regression.train import _prepare_images as prepare_face
from study.exp2_face_history_head32_regression import config as fusion_config
from study.exp2_face_history_head32_regression.models import build_pretrained_model as build_fusion
from study.exp6_face_pair_lab_delta import config as pair_config
from study.exp6_face_pair_lab_delta.data import PairedFrameDataset
from study.exp6_face_pair_lab_delta.models import build_model as build_pair
from study.exp6_face_pair_lab_delta.train import _prepare as prepare_pair


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXP_DIR / "outputs"
SOURCES = ("exp6_pair_delta", "exp2_face", "exp2_face_history")
SEED = 20260924
PERMUTATIONS = 999
_GPU = None


def _hash_key(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_per_patient(records, id_column):
    records = records.copy()
    records["selection_key"] = records[id_column].map(_hash_key)
    records = records.sort_values(["hospital_id", "selection_key", id_column])
    selected = records.drop_duplicates("hospital_id").drop(columns="selection_key")
    if selected.hospital_id.duplicated().any():
        raise AssertionError("Patient repeated in selected observations")
    return selected.sort_values("hospital_id").reset_index(drop=True)


def _load_selection(target):
    pair_path = pair_config.OUTPUT_DIR / "task_records" / f"{target}.csv"
    pair = pd.read_csv(pair_path, dtype={"hospital_id": str})
    pair = pair.loc[pair.split.eq("test")].copy()
    if pair.groupby("hospital_id").split.nunique().max() != 1:
        raise AssertionError(f"Exp6 patient split leakage: {target}")
    if not np.allclose(pair.raw_delta, pair.second_value - pair.first_value):
        raise AssertionError(f"Exp6 pair target mismatch: {target}")
    pair = _one_per_patient(pair, "pair_id")

    face_path = Path(face_config.OUTPUT_DIR) / "task_records" / f"{target}.csv"
    history_path = Path(fusion_config.OUTPUT_DIR) / "task_records" / f"{target}.csv"
    if not face_path.exists() or not history_path.exists():
        return {"exp6_pair_delta": pair}
    face = pd.read_csv(face_path, dtype={"hospital_id": str})
    history = pd.read_csv(history_path, dtype={"hospital_id": str})
    for name, frame in (("face", face), ("history", history)):
        if frame.groupby("hospital_id").split.nunique().max() != 1:
            raise AssertionError(f"Exp2 {name} patient split leakage: {target}")
        if frame.video_id.duplicated().any():
            raise AssertionError(f"Exp2 {name} duplicate video: {target}")
    face = face.loc[face.split.eq("test")].copy()
    history = history.loc[history.split.eq("test")].copy()
    shared = face.merge(
        history[["hospital_id", "video_id", "split", "raw_value"]],
        on=["hospital_id", "video_id", "split"],
        how="inner", suffixes=("", "_history"), validate="one_to_one",
    )
    if not np.allclose(shared.raw_value, shared.raw_value_history, rtol=0, atol=1e-8):
        raise AssertionError(f"Exp2 target values disagree: {target}")
    shared = _one_per_patient(shared.drop(columns="raw_value_history"), "video_id")
    return {
        "exp6_pair_delta": pair,
        "exp2_face": shared,
        "exp2_face_history": shared,
    }


def _worker_init(gpu_queue):
    global _GPU
    _GPU = int(gpu_queue.get())
    torch.cuda.set_device(_GPU)
    torch.set_num_threads(2)


def _loader(dataset):
    return DataLoader(
        dataset, batch_size=24, shuffle=False, num_workers=2,
        pin_memory=True, prefetch_factor=2,
    )


def _features(model, images):
    backbone = model.backbone if hasattr(model, "backbone") else model
    x = backbone.features(images)
    return backbone.avgpool(x).flatten(1).float()


@torch.inference_mode()
def _extract_face(model, frame_index, records, device):
    dataset = AllFramesDataset(frame_index, records, views=("original",), interpolation="bicubic")
    sums = np.zeros((len(records), 1280), dtype=np.float64)
    counts = np.zeros(len(records), dtype=np.int64)
    for images, _, frame_rows, codes in _loader(dataset):
        x = prepare_face(images, codes, "bicubic", device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = _features(model, x)
        video_rows = dataset.frame_video_rows[frame_rows.numpy()]
        np.add.at(sums, video_rows, features.cpu().numpy())
        np.add.at(counts, video_rows, 1)
    dataset.close()
    if not np.all(counts == 20):
        raise AssertionError(f"Expected 20 frames/video, got {np.unique(counts)}")
    return (sums / counts[:, None]).astype(np.float32)


@torch.inference_mode()
def _extract_pairs(model, frame_index, records, device):
    dataset = PairedFrameDataset(frame_index, records)
    first_sum = np.zeros((len(records), 1280), dtype=np.float64)
    second_sum = np.zeros_like(first_sum)
    counts = np.zeros(len(records), dtype=np.int64)
    for first, second, _, frame_rows, codes, _ in _loader(dataset):
        first_x = prepare_pair(first, codes, device)
        second_x = prepare_pair(second, codes, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = _features(model, torch.cat((first_x, second_x), dim=0))
        first_feat, second_feat = features.chunk(2)
        pair_rows = dataset.frame_pair_rows[frame_rows.numpy()]
        np.add.at(first_sum, pair_rows, first_feat.cpu().numpy())
        np.add.at(second_sum, pair_rows, second_feat.cpu().numpy())
        np.add.at(counts, pair_rows, 1)
    dataset.close()
    if not np.all(counts == 20):
        raise AssertionError(f"Expected 20 aligned frames/pair, got {np.unique(counts)}")
    return ((second_sum - first_sum) / counts[:, None]).astype(np.float32)


def _run_job(job):
    source, target = job["source"], job["target"]
    device = torch.device(f"cuda:{_GPU}")
    records = _load_selection(target)[source]
    if len(records) < 5:
        raise ValueError(f"Too few test patients: {source}/{target}")
    if source == "exp6_pair_delta":
        checkpoint = pair_config.OUTPUT_DIR / "runs" / target / "model.pt"
        index = FrameOffsetIndex.load(pair_config.CACHE_DIR / "frame_offsets.npz")
        model, _ = build_pair("shared")
    elif source == "exp2_face":
        checkpoint = Path(face_config.OUTPUT_DIR) / "runs" / "efficientnet_b0" / target / "model.pt"
        index = FrameOffsetIndex.load(Path(face_config.REFERENCE_INDEX_DIR) / "frame_offsets.npz")
        model, _, _ = build_face("efficientnet_b0", face_config.WEIGHTS_DIR)
    else:
        checkpoint = Path(fusion_config.OUTPUT_DIR) / "runs" / "efficientnet_b0" / target / "model.pt"
        index = FrameOffsetIndex.load(Path(fusion_config.INDEX_DIR) / "frame_offsets.npz")
        model, _, _ = build_fusion("efficientnet_b0", fusion_config.WEIGHTS_DIR)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["target"] != target:
        raise AssertionError(f"Checkpoint target mismatch: {checkpoint}")
    if source == "exp6_pair_delta" and state["model_variant"] != "shared":
        raise AssertionError(f"Expected shared-backbone Exp6: {checkpoint}")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.to(device, memory_format=torch.channels_last).eval()
    if source == "exp6_pair_delta":
        features = _extract_pairs(model, index, records, device)
        labels = records.raw_delta.to_numpy(np.float64)
        ids = records.pair_id.to_numpy(str)
    else:
        features = _extract_face(model, index, records, device)
        labels = records.raw_value.to_numpy(np.float64)
        ids = records.video_id.to_numpy(str)
    path = OUTPUT_DIR / "features" / source / f"{target}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, features=features, labels=labels, ids=ids,
        hospital_ids=records.hospital_id.to_numpy(str),
    )
    return {
        "source": source, "target": target, "patients": len(records),
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "features_path": str(path), "gpu": _GPU,
    }


def _association(features, labels, permutations=PERMUTATIONS, seed=SEED, strata=None):
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("Non-finite features or labels")
    lengths = np.linalg.norm(features, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("Zero-length feature representation")
    normalized = features / lengths[:, None]
    upper = np.triu_indices(len(labels), 1)
    cosine = (normalized @ normalized.T)[upper].astype(np.float64)
    gaps = np.abs(labels[:, None] - labels[None, :])[upper]
    feature_rank = rankdata(cosine)
    feature_rank -= feature_rank.mean()
    feature_rank /= np.linalg.norm(feature_rank)

    def rho(values):
        ranks = rankdata(-np.abs(values[:, None] - values[None, :])[upper])
        ranks -= ranks.mean()
        norm = np.linalg.norm(ranks)
        return float(feature_rank @ (ranks / norm)) if norm > 0 else np.nan

    observed = rho(labels)
    rng = np.random.default_rng(seed)
    null = np.asarray([rho(rng.permutation(labels)) for _ in range(permutations)])
    valid_null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(valid_null):
        raise ValueError("Degenerate label-distance distribution")
    p_value = (1 + int(np.sum(valid_null >= observed))) / (1 + len(valid_null))
    conditional_p = np.nan
    if strata is not None:
        strata = np.asarray(strata)
        if len(strata) != len(labels):
            raise ValueError("Interval strata and labels have different lengths")
        groups = [np.flatnonzero(strata == group) for group in np.unique(strata)]
        conditional_null = []
        for _ in range(permutations):
            shuffled = labels.copy()
            for indices in groups:
                shuffled[indices] = rng.permutation(labels[indices])
            conditional_null.append(rho(shuffled))
        conditional_null = np.asarray(conditional_null)
        conditional_null = conditional_null[np.isfinite(conditional_null)]
        if len(conditional_null):
            conditional_p = (1 + int(np.sum(conditional_null >= observed))) / (1 + len(conditional_null))
    bins = pd.qcut(gaps, q=5, labels=False, duplicates="drop")
    curve = pd.DataFrame({"bin": bins, "cosine": cosine, "gap": gaps})
    curve = curve.groupby("bin", as_index=False).agg(
        mean_cosine=("cosine", "mean"), median_gap=("gap", "median"),
        pair_count=("cosine", "size"),
    )
    return {
        "rho": observed, "permutation_p_one_sided": p_value,
        "interval_stratified_p_one_sided": conditional_p,
        "null_rho_low": float(np.quantile(valid_null, 0.025)),
        "null_rho_high": float(np.quantile(valid_null, 0.975)),
        "cosine_near_minus_far": float(curve.mean_cosine.iloc[0] - curve.mean_cosine.iloc[-1]),
        "pair_count": len(cosine), "curve": curve,
    }


def _fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranks = np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def _plot(metrics, curves):
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 170})
    figures = OUTPUT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    targets = list(pair_config.TARGETS)
    matrix = metrics.pivot(index="source", columns="target", values="rho").reindex(
        index=SOURCES, columns=targets
    )
    significance = metrics.pivot(index="source", columns="target", values="fdr_q").reindex(
        index=SOURCES, columns=targets
    )
    fig, ax = plt.subplots(figsize=(16, 4.2), layout="constrained")
    limit = max(0.2, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(targets)), [name.replace("_", "\n") for name in targets])
    ax.set_yticks(range(len(SOURCES)), [
        "Exp6 pair delta", "Exp2 face", "Exp2 face+history (face branch)"
    ])
    for row in range(len(SOURCES)):
        for col in range(len(targets)):
            value = matrix.iloc[row, col]
            if np.isfinite(value):
                marker = "*" if significance.iloc[row, col] < 0.05 else ""
                ax.text(col, row, f"{value:+.2f}{marker}", ha="center", va="center",
                        color="white" if abs(value) > limit * 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Spearman rho: cosine similarity vs negative lab gap")
    ax.set_title("Held-out, one observation per patient; * global-permutation FDR q < 0.05")
    fig.savefig(figures / "association_overview.png")
    plt.close(fig)

    for source in SOURCES:
        subset = metrics.loc[metrics.source.eq(source)].set_index("target")
        fig, axes = plt.subplots(3, 3, figsize=(13, 10), layout="constrained")
        for ax, target in zip(axes.flat, targets):
            if target not in subset.index:
                ax.axis("off")
                continue
            curve = curves[(source, target)]
            x = np.arange(1, len(curve) + 1)
            ax.plot(x, curve.mean_cosine, marker="o", linewidth=1.6)
            ax.set_xticks(x)
            ax.grid(alpha=0.25)
            row = subset.loc[target]
            ax.set_title(f"{target}\nn={int(row.patients)}; rho={row.rho:+.2f}; q={row.fdr_q:.3f}", fontsize=10)
            ax.set_xlabel("Lab-gap quantile (near to far)")
            ax.set_ylabel("Mean feature cosine")
        title = (
            "Exp2 face+history: face encoder only"
            if source == "exp2_face_history" else source.replace("_", " ")
        )
        fig.suptitle(title + " | original-view 20-frame means", fontsize=14)
        fig.savefig(figures / f"label_gap_curves_{source}.png")
        plt.close(fig)

    paired = matrix.loc[["exp2_face", "exp2_face_history"]].dropna(axis=1)
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    for target in paired.columns:
        x, y = paired.loc["exp2_face", target], paired.loc["exp2_face_history", target]
        ax.scatter(x, y, s=55, label=target)
    limits = [min(-0.05, paired.min().min() - 0.05), max(0.05, paired.max().max() + 0.05)]
    ax.plot(limits, limits, "--", color="0.5", linewidth=1)
    ax.set(xlim=limits, ylim=limits, xlabel="Exp2 face-only rho", ylabel="Exp2 face + history face-encoder rho",
           title="Same held-out patient/video per target")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.savefig(figures / "exp2_same_cohort_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()
    targets = list(pair_config.TARGETS)
    jobs = []
    for target in targets:
        available = _load_selection(target)
        for source in available:
            jobs.append({"source": source, "target": target})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_extraction:
        gpu_ids = [int(value) for value in args.gpus.split(",")]
        context = mp.get_context("spawn")
        with context.Manager() as manager:
            queue = manager.Queue()
            for gpu in gpu_ids:
                queue.put(gpu)
            with ProcessPoolExecutor(
                max_workers=len(gpu_ids), mp_context=context,
                initializer=_worker_init, initargs=(queue,),
            ) as executor:
                futures = {executor.submit(_run_job, job): job for job in jobs}
                completed = []
                for future in as_completed(futures):
                    result = future.result()
                    completed.append(result)
                    print(f"[features] {result['source']}/{result['target']} "
                          f"patients={result['patients']} gpu={result['gpu']}", flush=True)
        pd.DataFrame(completed).sort_values(["source", "target"]).to_csv(
            OUTPUT_DIR / "feature_manifest.csv", index=False
        )
    metrics, curves = [], {}
    for job in jobs:
        source, target = job["source"], job["target"]
        with np.load(OUTPUT_DIR / "features" / source / f"{target}.npz") as archive:
            features = archive["features"].astype(np.float64)
            labels = archive["labels"].astype(np.float64)
            patients = archive["hospital_ids"]
            ids = archive["ids"]
        if len(set(patients)) != len(patients):
            raise AssertionError(f"Duplicate patients in {source}/{target}")
        if source == "exp2_face_history":
            with np.load(OUTPUT_DIR / "features" / "exp2_face" / f"{target}.npz") as face:
                if not np.array_equal(ids, face["ids"]) or not np.array_equal(labels, face["labels"]):
                    raise AssertionError(f"Exp2 cohorts disagree for {target}")
        strata = None
        if source == "exp6_pair_delta":
            records = _load_selection(target)[source]
            if not np.array_equal(ids, records.pair_id.to_numpy(str)) or not np.allclose(labels, records.raw_delta):
                raise AssertionError(f"Exp6 saved features no longer match task records: {target}")
            intervals = records.lab_interval_h.to_numpy(np.float64)
            strata = pd.qcut(intervals, q=min(4, len(intervals)), labels=False, duplicates="drop")
        result = _association(features, labels, args.permutations,
                              SEED + int(_hash_key(f"{source}:{target}")[:8], 16),
                              strata=strata)
        curves[(source, target)] = result.pop("curve")
        metrics.append({"source": source, "target": target, "patients": len(labels), **result})
        print(f"[association] {source}/{target} rho={result['rho']:+.3f} "
              f"p={result['permutation_p_one_sided']:.4f}", flush=True)
    metrics = pd.DataFrame(metrics).sort_values(["source", "target"]).reset_index(drop=True)
    metrics["fdr_q"] = _fdr_bh(metrics.permutation_p_one_sided)
    pair_mask = metrics.source.eq("exp6_pair_delta")
    metrics.loc[pair_mask, "interval_stratified_fdr_q"] = _fdr_bh(
        metrics.loc[pair_mask, "interval_stratified_p_one_sided"]
    )
    metrics.to_csv(OUTPUT_DIR / "association_metrics.csv", index=False)
    pd.concat([
        curve.assign(source=source, target=target) for (source, target), curve in curves.items()
    ], ignore_index=True).to_csv(OUTPUT_DIR / "gap_quantile_curves.csv", index=False)
    _plot(metrics, curves)
    (OUTPUT_DIR / "method.json").write_text(json.dumps({
        "schema_version": 1, "test_only": True, "one_observation_per_patient": True,
        "selection": "minimum SHA256 of fixed seed and observation ID; no label-based selection",
        "exp2_same_cohort": "intersection of test video IDs and identical raw labels",
        "exp6_model": "original shared-backbone five-view trained checkpoint",
        "frame_policy": "mean encoder features over all 20 indexed frames, original view",
        "exp6_representation": "mean(late-frame features) - mean(early-frame features)",
        "exp2_representation": "mean backbone pooled face feature, excluding history encoder",
        "similarity": "cosine between L2-normalized video or pair representations",
        "label_gap": "absolute difference of raw lab values or raw lab deltas",
        "test": "Spearman rank correlation with negative lab gap; patient-label permutation, one-sided",
        "permutations": args.permutations, "multiple_testing": "Benjamini-Hochberg across all available model-task tests",
        "exp6_interval_sensitivity": "Additional patient-label permutations within quartiles of the pair's elapsed lab time; BH correction across nine Exp6 targets",
        "caution": "Associational; Exp6 and Exp2 splits differ, so their rho values are not paired cohort comparisons.",
    }, indent=2), encoding="utf-8")
    print(f"[complete] tests={len(metrics)} output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
