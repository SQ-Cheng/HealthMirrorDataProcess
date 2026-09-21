"""Validate the controlled three-pathway experiment and regenerate final figures."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .config import TARGETS


ROOT = Path(__file__).resolve().parents[2]
FACE_HISTORY_DIR = ROOT / "study/exp2_face_history_head32_regression/outputs/20frame"
FACE_ONLY_DIR = ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame"
HISTORY_ONLY_DIR = ROOT / "study/exp2_history_only_head32_regression/outputs"
EXPECTED_TARGETS = tuple(TARGETS)
ALIGNMENT_COLUMNS = (
    "hospital_id",
    "video_id",
    "binary_label",
    "raw_value",
    "score_threshold",
    "score_scale",
    "robust_scaled_raw_value",
    "split",
)


def _validate_image_experiment(output_dir):
    run_index = pd.read_csv(output_dir / "run_index.csv")
    expected_jobs = {("efficientnet_b0", target) for target in EXPECTED_TARGETS}
    actual_jobs = set(
        run_index[["architecture", "target"]].itertuples(index=False, name=None)
    )
    if actual_jobs != expected_jobs or len(run_index) != len(expected_jobs):
        raise RuntimeError(f"Unexpected jobs in {output_dir}: {actual_jobs}")
    if not run_index["status"].eq("ok").all():
        raise RuntimeError(f"Failed jobs remain in {output_dir}")
    for row in run_index.itertuples(index=False):
        checkpoint = Path(row.run_dir) / "model.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    metric_keys = set(
        metrics[["architecture", "target", "split"]].itertuples(
            index=False, name=None
        )
    )
    expected_metrics = {
        ("efficientnet_b0", target, split)
        for target in EXPECTED_TARGETS
        for split in ("train", "val", "test")
    }
    if metric_keys != expected_metrics or len(metrics) != len(expected_metrics):
        raise RuntimeError(f"Incomplete or duplicate metrics in {output_dir}")
    failures = pd.read_csv(output_dir / "failures.csv")
    if not failures.empty:
        raise RuntimeError(f"Non-empty failures.csv in {output_dir}")
    return {"jobs": len(run_index), "metric_rows": len(metrics)}


def _validate_history_only(output_dir):
    run_index = pd.read_csv(output_dir / "run_index.csv")
    actual_targets = set(run_index["target"].astype(str))
    if actual_targets != set(EXPECTED_TARGETS) or len(run_index) != len(EXPECTED_TARGETS):
        raise RuntimeError(f"Unexpected history-only targets: {actual_targets}")
    if not run_index["status"].eq("ok").all():
        raise RuntimeError("Failed history-only jobs remain")
    for row in run_index.itertuples(index=False):
        checkpoint = Path(row.run_dir) / "model.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Missing checkpoint: {checkpoint}")
    metrics = pd.read_csv(output_dir / "metrics_all.csv")
    expected_metrics = {
        ("history_only_head32", target, split)
        for target in EXPECTED_TARGETS
        for split in ("train", "val", "test")
    }
    actual_metrics = set(
        metrics[["architecture", "target", "split"]].itertuples(
            index=False, name=None
        )
    )
    if actual_metrics != expected_metrics or len(metrics) != len(expected_metrics):
        raise RuntimeError("Incomplete or duplicate history-only metrics")
    failures = pd.read_csv(output_dir / "failures.csv")
    if not failures.empty:
        raise RuntimeError("Non-empty history-only failures.csv")
    audit = pd.read_csv(output_dir / "data_alignment_audit.csv")
    if audit.empty or not audit["exact_match"].astype(bool).all():
        raise RuntimeError("History-only reference alignment failed")
    return {"jobs": len(run_index), "metric_rows": len(metrics)}


def _validate_face_alignment():
    for target in EXPECTED_TARGETS:
        history = pd.read_csv(
            FACE_HISTORY_DIR / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str},
        ).sort_values("video_id").reset_index(drop=True)
        face = pd.read_csv(
            FACE_ONLY_DIR / "task_records" / f"{target}.csv",
            dtype={"hospital_id": str, "video_id": str},
        ).sort_values("video_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            history[list(ALIGNMENT_COLUMNS)],
            face[list(ALIGNMENT_COLUMNS)],
            check_dtype=True,
            check_exact=True,
        )
        for frame in (history, face):
            if frame.groupby("hospital_id")["split"].nunique().gt(1).any():
                raise RuntimeError(f"Patient leakage for {target}")
    history_scalers = json.loads(
        (FACE_HISTORY_DIR / "target_scalers.json").read_text(encoding="utf-8")
    )["targets"]
    face_scalers = json.loads(
        (FACE_ONLY_DIR / "target_scalers.json").read_text(encoding="utf-8")
    )["targets"]
    if history_scalers != face_scalers:
        raise RuntimeError("Face-only target scalers differ from face+history")
    return {"targets": len(EXPECTED_TARGETS), "exact_alignment": True}


def _generate_plots():
    from .plot_results import main as plot_face_history
    from study.exp2_face_pretrained_head32_regression.plot_results import (
        main as plot_face_only,
    )
    from study.exp2_history_only_head32_regression.plot_results import (
        main as plot_history_only,
    )

    plot_face_history(FACE_HISTORY_DIR)
    plot_face_only(FACE_ONLY_DIR)
    plot_history_only(HISTORY_ONLY_DIR, FACE_HISTORY_DIR, FACE_ONLY_DIR)
    required = (
        FACE_HISTORY_DIR / "figures/test_metrics.png",
        FACE_ONLY_DIR / "figures/test_metrics.png",
        HISTORY_ONLY_DIR / "figures/three_pathway_model_comparison.png",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing final figure: {path}")
    return [str(path.resolve()) for path in required]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("face_history", "face_only", "history_only", "all"),
        required=True,
    )
    parser.add_argument("--generate-plots", action="store_true")
    args = parser.parse_args()
    audit = {"schema_version": 1, "stage": args.stage}
    if args.stage in ("face_history", "all"):
        audit["face_history"] = _validate_image_experiment(FACE_HISTORY_DIR)
    if args.stage in ("face_only", "all"):
        audit["face_only"] = _validate_image_experiment(FACE_ONLY_DIR)
        audit["face_alignment"] = _validate_face_alignment()
    if args.stage in ("history_only", "all"):
        audit["history_only"] = _validate_history_only(HISTORY_ONLY_DIR)
    if args.generate_plots:
        if args.stage != "all":
            raise ValueError("--generate-plots requires --stage all")
        audit["figures"] = _generate_plots()
    audit["validated_at_utc"] = datetime.now(timezone.utc).isoformat()
    destination = (
        HISTORY_ONLY_DIR / "three_experiment_completion_audit.json"
        if args.stage == "all"
        else ROOT / "study/exp2_face_history_head32_regression/logs" /
        f"validation_{args.stage}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
