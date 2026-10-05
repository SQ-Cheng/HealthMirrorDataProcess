"""Remove superseded pixel-dependent artifacts without breaking 128-only studies."""

import argparse
import fcntl
import json
from pathlib import Path
import shutil

import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
STATE = STUDY / "common/outputs/face224_reruns"
AUDIT = STUDY / "common/outputs/retire_128"
REG = STUDY / "exp2_face_pretrained_head32_regression"
CLASS = STUDY / "exp2_face_pretrained_head32_classification"
DELTA = STUDY / "exp6_face_pair_lab_delta"
SPECTRAL = STUDY / "exp8_rgb_spectral_lab_regression"
CLINICAL = {REG / "outputs/20frame": {"task_records", "source_data", "run_index.csv", "target_scalers.json"},
            REG / "outputs/5fold": {"splits", "CLINICAL_REFERENCE_ONLY.json"},
            DELTA / "outputs": {"task_records", "source_data", "run_index.csv", "target_scalers.json", "experiment_manifest.json"}}


def bytes_in(path):
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def native(name):
    return "face224" in name


def candidates(final):
    paths = []
    protected_queue = {"lab_match_6h", "lab_match_12h"}
    if not final:
        protected_queue.add("patient_diverse_schedule_30_40")
    for root in (REG, CLASS, DELTA):
        # Native data is explicitly named; never recurse destructively into a
        # mixed directory such as outputs/ablations.
        output = root / "outputs"
        for child in output.iterdir():
            if native(child.name) or child in CLINICAL:
                continue
            if root == DELTA and child.name == "runs":
                continue
            if child.name == "ablations":
                paths.extend(p for p in child.iterdir() if not native(p.name)
                             and (final or p.name not in protected_queue))
            elif final or child.name not in {"runs", "run_index.csv", "metrics_all.csv", "target_scalers.json"}:
                if root == DELTA and child.name in CLINICAL[DELTA / "outputs"]:
                    continue
                paths.append(child)
        logs = root / "logs"
        if logs.exists():
            for child in logs.iterdir():
                if native(child.name):
                    continue
                if child.name == "ablations":
                    paths.extend(p for p in child.iterdir() if not native(p.name))
                else:
                    paths.append(child)
        # Exp6's old index is still a direct input to its protected
        # direction-classification main experiment.
        if root != DELTA and (root / "cache").exists():
            paths.extend(p for p in (root / "cache").iterdir() if not native(p.name))
    for baseline, keep in CLINICAL.items():
        if final:
            paths.extend(p for p in baseline.iterdir() if p.name not in keep
                         and not native(p.name) and p.name != "ablations" and p.name != "runs")
        runs = baseline / "runs"
        if runs.exists():
            for path in runs.rglob("*"):
                if path.is_file() and (final or path.suffix in {".pt", ".pth"}):
                    if baseline == DELTA / "outputs" and path.name == "pair_predictions.csv":
                        continue
                    paths.append(path)
    # If a new Exp8 main has not completed, it remains a protected 128-only
    # experiment. Do not delete its inputs, weights or results prematurely.
    if final and all((SPECTRAL / f"outputs{suffix}_face224/COMPLETE").is_file() for suffix in ("", "_grid16")):
        paths.extend(SPECTRAL / name for name in ("outputs", "outputs_grid16", "cache", "cache_grid16", "logs", "logs_grid16") if (SPECTRAL / name).exists())
    # Comparisons containing deleted legacy predictions are superseded too.
    if final:
        for root in (REG, CLASS, DELTA, SPECTRAL):
            for path in root.rglob("face224*.*"):
                if "/outputs" in str(path) and (path.name.startswith("face224_vs_legacy_")
                        or path.name in {"face224_comparison.csv", "face224_common_test_predictions.csv"}):
                    paths.append(path)
        paths.extend(STATE / name for name in ("comparison_all.csv", "REPORT.md") if (STATE / name).exists())
    unique = sorted(set(paths), key=lambda p: (len(p.parts), str(p)))
    selected = []
    for path in unique:
        if path.exists() and not any(parent in selected for parent in path.parents):
            if STUDY not in path.resolve().parents or path.is_symlink():
                raise ValueError(f"Unsafe deletion target: {path}")
            selected.append(path)
    return selected


def write_policy():
    migrated = {REG.name, CLASS.name, DELTA.name, "exp2_face224_lab_change_tracking"}
    if all((SPECTRAL / f"outputs{suffix}_face224/COMPLETE").is_file() for suffix in ("", "_grid16")):
        migrated.add(SPECTRAL.name)
    protected = sorted(p.name for p in STUDY.iterdir() if p.is_dir()
                       and p.name.startswith("exp") and p.name not in migrated)
    policy = {
        "protected_studies": protected,
        "external_raw_or_legacy_video_directories_deleted": False,
        "protected_dependencies": [str(p.relative_to(STUDY)) for p in CLINICAL] + [str((DELTA / "cache/frames20").relative_to(STUDY))],
        "shared_legacy_decoder": "Retained only for protected 128-only studies; migrated Exp2/Exp6 entry points reject legacy input",
        "removed_bytes": int(sum(pd.read_csv(p).bytes.sum() for p in AUDIT.glob("removed_*.csv"))),
    }
    import hashlib
    split_manifest = REG / "outputs/5fold/splits/manifest.json"
    if split_manifest.exists():
        policy["protected_exp3_split_sha256"] = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "policy.json").write_text(json.dumps(policy, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.final:
        with (STATE / ".queue.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The active migration queue still needs legacy clinical baselines")
            if not (STATE / "COMPLETE").is_file():
                from study.common.rerun_face224 import experiment_plan, preflight
                plan = experiment_plan()
                if any("baseline" in job for job in plan):
                    raise RuntimeError("An incomplete legacy-dependent queue cannot be cleaned")
                preflight(plan)
    rows = [{"path": str(p.relative_to(STUDY.parent)), "bytes": bytes_in(p),
             "reason": "superseded 128 pixel-dependent artifact"} for p in candidates(args.final)]
    print(f"paths={len(rows)} bytes={sum(row['bytes'] for row in rows)} final={args.final} apply={args.apply}", flush=True)
    if not args.apply:
        return
    AUDIT.mkdir(parents=True, exist_ok=True)
    phase = "final" if args.final else "unused"
    pd.DataFrame(rows).to_csv(AUDIT / f"removed_{phase}.csv", index=False)
    for row in rows:
        path = STUDY.parent / row["path"]
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if args.final:
        for root in CLINICAL:
            (root / "CLINICAL_REFERENCE_ONLY.json").write_text(json.dumps({
                "contains_128_video_pixels_or_models": False,
                "reason": "Immutable clinical labels/splits required by protected 128-only Exp3 and Exp6 direction classification",
                "not_a_training_result": True,
            }, indent=2))
            runs = root / "runs"
            if runs.exists():
                for directory in sorted(runs.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if directory.is_dir() and not any(directory.iterdir()):
                        directory.rmdir()
                if not any(runs.iterdir()):
                    runs.rmdir()
        write_policy()


if __name__ == "__main__":
    main()
