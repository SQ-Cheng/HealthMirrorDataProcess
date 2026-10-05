"""Evaluate held-out longitudinal changes from saved native-224 predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
OUTPUT = ROOT / "outputs"
ANALYTES = {
    "oxyhemoglobin_fraction": ("Oxyhemoglobin fraction", "oxyhemoglobin_fraction", "%"),
    "lactate_high": ("Lactate", "lactate", "mmol/L"),
    "urea_high": ("Urea", "urea", "mmol/L"),
    "total_bilirubin_high": ("Total bilirubin", "total_bilirubin", "umol/L"),
    "platelet_count_low": ("Platelet count", "platelet_count", "10^9/L"),
    "hemoglobin_low": ("Hemoglobin", "hemoglobin", "g/L"),
    "aa_po2_ratio_low": ("A/a PO2 ratio", "aa_po2_ratio", "%"),
    "creatinine_high": ("Creatinine", "creatinine", "umol/L"),
}


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def select_events(frame):
    # Select without inspecting values or predictions; duplicate lab matches
    # must not create artificial repeated longitudinal observations.
    ranked = frame.sort_values(["hospital_id", "lab_time_unix", "match_delta_h",
                                "midpoint_distance_h", "video_id"])
    counts = ranked.groupby(["hospital_id", "lab_time_unix"]).size().rename("matched_videos")
    chosen = ranked.drop_duplicates(["hospital_id", "lab_time_unix"]).copy()
    chosen = chosen.join(counts, on=["hospital_id", "lab_time_unix"])
    consistency = ranked.groupby(["hospital_id", "lab_time_unix"]).y_true.agg(["min", "max"])
    if not np.allclose(consistency["min"], consistency["max"], rtol=0, atol=1e-6):
        raise ValueError("One lab event has conflicting values")
    return chosen.reset_index(drop=True)


def adjacent_pairs(events):
    rows = []
    for patient, group in events.groupby("hospital_id", sort=True):
        ordered = group.sort_values("lab_time_unix").to_dict("records")
        for first, second in zip(ordered, ordered[1:]):
            rows.append({
                "hospital_id": patient,
                "first_video_id": first["video_id"], "second_video_id": second["video_id"],
                "first_lab_time_unix": first["lab_time_unix"], "second_lab_time_unix": second["lab_time_unix"],
                "first_capture_time_unix": first["capture_time_unix"], "second_capture_time_unix": second["capture_time_unix"],
                "lab_interval_h": (second["lab_time_unix"] - first["lab_time_unix"]) / 3600,
                "video_interval_h": (second["capture_time_unix"] - first["capture_time_unix"]) / 3600,
                "first_match_delta_h": first["match_delta_h"], "second_match_delta_h": second["match_delta_h"],
                "first_true": first["y_true"], "second_true": second["y_true"],
                "first_pred": first["y_pred"], "second_pred": second["y_pred"],
                "true_delta": second["y_true"] - first["y_true"],
                "pred_delta": second["y_pred"] - first["y_pred"],
                "included": second["capture_time_unix"] > first["capture_time_unix"],
            })
    return pd.DataFrame(rows)


def pair_metrics(frame):
    x, y = frame.true_delta.to_numpy(), frame.pred_delta.to_numpy()
    active = np.abs(x) > 1e-6
    truth, prediction = np.sign(x[active]), np.where(np.abs(y[active]) <= 1e-6, 0, np.sign(y[active]))
    recalls = [np.mean(prediction[truth == side] == side) for side in (-1, 1) if np.any(truth == side)]
    sst = np.sum((x - x.mean()) ** 2)
    persistence_mae = np.mean(np.abs(x))
    return {
        "delta_r": correlation(x, y),
        "delta_spearman": float(spearmanr(x, y).statistic) if len(x) >= 3 and np.ptp(x) and np.ptp(y) else np.nan,
        "delta_mae": float(np.mean(np.abs(y - x))),
        "delta_rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "delta_r2": float(1 - np.sum((y - x) ** 2) / sst) if sst > 1e-12 else np.nan,
        "persistence_mae": float(persistence_mae),
        "mae_skill_vs_no_change": float(1 - np.mean(np.abs(y - x)) / persistence_mae) if persistence_mae > 1e-12 else np.nan,
        "direction_accuracy": float(np.mean(truth == prediction)) if len(truth) else np.nan,
        "direction_bacc": float(np.mean(recalls)) if len(recalls) == 2 else np.nan,
    }


def clustered_intervals(frame, iterations, rng):
    groups = [group for _, group in frame.groupby("hospital_id")]
    samples = {key: [] for key in ("delta_r", "direction_bacc", "mae_skill_vs_no_change")}
    for _ in range(iterations):
        draw = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))], ignore_index=True)
        for key, value in pair_metrics(draw).items():
            if key in samples and np.isfinite(value):
                samples[key].append(value)
    result = {}
    for key, values in samples.items():
        bounds = np.quantile(values, [.025, .975]) if len(values) >= max(20, iterations // 2) else [np.nan, np.nan]
        result.update({f"{key}_ci_low": bounds[0], f"{key}_ci_high": bounds[1], f"{key}_bootstrap_valid": len(values)})
    return result


def read_test(source, target):
    prefix = ANALYTES[target][1]
    run = source / f"runs/efficientnet_b0/{target}/video_predictions.csv"
    predictions = pd.read_csv(run, dtype={"hospital_id": str, "video_id": str})
    records = pd.read_csv(source / f"task_records/{target}.csv", dtype={"hospital_id": str, "video_id": str})
    if records.groupby("hospital_id").split.nunique().max() != 1:
        raise ValueError("Patient leakage in saved split")
    expected = records.loc[records.split.eq("test")].copy()
    predictions = predictions.loc[predictions.split.eq("test")].copy()
    if set(expected.video_id) != set(predictions.video_id) or predictions.video_id.duplicated().any():
        raise ValueError("Incomplete or duplicated test predictions")
    joint = predictions.merge(expected[["video_id", "hospital_id", "raw_value", "match_delta_h"]],
                              on=["video_id", "hospital_id"], validate="one_to_one", suffixes=("", "_record"))
    np.testing.assert_allclose(joint.y_true, joint.raw_value_record, rtol=0, atol=1e-6)
    if not joint.frame_count.eq(20).all():
        raise ValueError("Expected exactly twenty evaluation frames")
    source_data = pd.read_csv(source / "source_data/base_manifest.csv", dtype={"hospital_id": str, "video_id": str})
    fields = ["hospital_id", "video_id", "capture_time_unix", "capture_start_unix", "capture_end_unix",
              f"{prefix}_value", f"{prefix}_lab_time_unix", f"{prefix}_delta_h"]
    joint = joint.merge(source_data[fields], on=["hospital_id", "video_id"], validate="one_to_one")
    np.testing.assert_allclose(joint.y_true, joint[f"{prefix}_value"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(joint.match_delta_h, joint[f"{prefix}_delta_h"], rtol=0, atol=1e-6)
    joint["lab_time_unix"] = joint[f"{prefix}_lab_time_unix"]
    joint["midpoint_distance_h"] = np.abs(joint.lab_time_unix - joint.capture_time_unix) / 3600
    distance = np.maximum(np.maximum(joint.capture_start_unix - joint.lab_time_unix,
                                     joint.lab_time_unix - joint.capture_end_unix), 0) / 3600
    np.testing.assert_allclose(distance, joint.match_delta_h, rtol=0, atol=1e-6)
    if distance.gt(24 + 1e-6).any() or not np.isfinite(joint[["y_true", "y_pred", "lab_time_unix", "capture_time_unix"]]).all().all():
        raise ValueError("Invalid finite labels/predictions or matching limit")
    return joint


def plot(summary, pairs, events, figures):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, (target, definition) in zip(axes.flat, ANALYTES.items()):
        group = pairs.loc[pairs.target.eq(target) & pairs.included]
        ax.scatter(group.true_delta, group.pred_delta, s=19, alpha=.65, color="#247b88", edgecolors="none")
        if len(group) and np.ptp(group.true_delta) > 0:
            grid = np.linspace(group.true_delta.min(), group.true_delta.max(), 100)
            slope, intercept = np.polyfit(group.true_delta, group.pred_delta, 1)
            ax.plot(grid, slope * grid + intercept, color="#b94d45", label="OLS fit")
            ax.plot(grid, grid, "--", color=".5", linewidth=1, label="Identity")
        row = summary.loc[summary.target.eq(target)].iloc[0]
        ax.set(title=f"{definition[0]}\nr={row.delta_r:.2f}; pairs={row.pairs}; patients={row.longitudinal_patients}",
               xlabel=f"Observed change ({definition[2]})", ylabel=f"Predicted change ({definition[2]})")
        ax.axhline(0, color=".8", linewidth=.7)
        ax.axvline(0, color=".8", linewidth=.7)
    axes.flat[0].legend(fontsize=8)
    fig.savefig(figures / "change_predicted_vs_observed.png", dpi=200)
    fig.savefig(figures / "change_predicted_vs_observed.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    positions = np.arange(len(summary))
    for ax, key, label, baseline in zip(axes, ("delta_r", "direction_bacc", "mae_skill_vs_no_change"),
                                      ("Change Pearson r", "Direction balanced accuracy", "MAE skill vs. no change"), (0, .5, 0)):
        estimates = summary[key].to_numpy()
        low, high = summary[f"{key}_ci_low"].to_numpy(), summary[f"{key}_ci_high"].to_numpy()
        ax.barh(positions, estimates, color="#247b88", alpha=.85)
        for pos, left, right in zip(positions, low, high):
            if np.isfinite(left) and np.isfinite(right):
                ax.plot([left, right], [pos, pos], color="black", linewidth=1.1)
        ax.axvline(baseline, color="#b94d45", linestyle="--", linewidth=1)
        ax.set(yticks=positions, yticklabels=[ANALYTES[t][0] for t in summary.target], xlabel=label)
        ax.invert_yaxis()
    fig.suptitle("Held-out within-patient changes: 95% patient-cluster bootstrap intervals")
    fig.savefig(figures / "tracking_performance.png", dpi=200)
    fig.savefig(figures / "tracking_performance.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, (target, definition) in zip(axes.flat, ANALYTES.items()):
        group = events.loc[events.target.eq(target)]
        sizes = group.groupby("hospital_id").size().sort_values(ascending=False, kind="stable")
        patient = sizes.index[0]
        selected = group.loc[group.hospital_id.eq(patient)].sort_values("lab_time_unix")
        days = (selected.lab_time_unix - selected.lab_time_unix.min()) / 86400
        ax.plot(days, selected.y_true, "o-", color="#247b88", label="Observed")
        ax.plot(days, selected.y_pred, "s--", color="#b94d45", label="Predicted")
        ax.set(title=f"{definition[0]}\nPatient {patient}; {len(selected)} lab events",
               xlabel="Days since first matched test lab", ylabel=f"Value ({definition[2]})")
    axes.flat[0].legend()
    fig.suptitle("Examples selected by largest event count, not prediction quality")
    fig.savefig(figures / "patient_trajectories.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 40:
        raise ValueError("At least forty bootstrap resamples are required")
    manifest = json.loads((args.source / "experiment_manifest.json").read_text())
    if manifest["hours"] != 24 or "face224" not in manifest["frame_index"] or not (args.source / "COMPLETE").exists():
        raise ValueError("A completed native-224 / 24h regression is required")
    tables, figures = args.output / "tables", args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261006)
    summaries, all_pairs, all_events = [], [], []
    signatures = {}
    for target, definition in ANALYTES.items():
        test = read_test(args.source, target)
        events = select_events(test)
        pairs = adjacent_pairs(events)
        if pairs.empty:
            raise ValueError(f"No longitudinal test observations: {target}")
        valid = pairs.loc[pairs.included]
        if valid.empty:
            raise ValueError(f"No ordered longitudinal test pairs: {target}")
        patient_ids = valid.hospital_id.unique()
        longitudinal = events.loc[events.hospital_id.isin(patient_ids)].copy()
        centered_true = longitudinal.y_true - longitudinal.groupby("hospital_id").y_true.transform("mean")
        centered_pred = longitudinal.y_pred - longitudinal.groupby("hospital_id").y_pred.transform("mean")
        active = valid.loc[valid.true_delta.abs().gt(1e-6)].copy()
        active["correct"] = (np.sign(active.true_delta) == np.sign(active.pred_delta)) & active.pred_delta.abs().gt(1e-6)
        row = {
            "target": target, "unit": definition[2], "test_videos": len(test), "test_patients": test.hospital_id.nunique(),
            "distinct_lab_events": len(events), "duplicate_video_matches_removed": len(test) - len(events),
            "longitudinal_patients": valid.hospital_id.nunique(), "pairs": len(valid),
            "order_excluded_pairs": int((~pairs.included).sum()),
            "rise_pairs": int(valid.true_delta.gt(1e-6).sum()), "fall_pairs": int(valid.true_delta.lt(-1e-6).sum()),
            "unchanged_pairs": int(valid.true_delta.abs().le(1e-6).sum()),
            "predicted_ties_on_changed_pairs": int(active.pred_delta.abs().le(1e-6).sum()),
            "patient_macro_direction_accuracy": active.groupby("hospital_id").correct.mean().mean(),
            "within_patient_centered_r": correlation(centered_true, centered_pred),
            **pair_metrics(valid), **clustered_intervals(valid, args.bootstrap, rng),
        }
        summaries.append(row)
        all_pairs.append(pairs.assign(target=target))
        all_events.append(events.assign(target=target))
        for relative in (f"runs/efficientnet_b0/{target}/video_predictions.csv", f"task_records/{target}.csv"):
            signatures[relative] = hashlib.sha256((args.source / relative).read_bytes()).hexdigest()
        print(f"{target}: patients={row['longitudinal_patients']} pairs={len(valid)} r={row['delta_r']:.3f} bACC={row['direction_bacc']:.3f}", flush=True)
    summary = pd.DataFrame(summaries)
    pairs, events = pd.concat(all_pairs, ignore_index=True), pd.concat(all_events, ignore_index=True)
    summary.to_csv(tables / "metrics.csv", index=False)
    pairs.to_csv(tables / "adjacent_lab_pairs.csv", index=False)
    events.to_csv(tables / "selected_lab_events.csv", index=False)
    signatures["source_data/base_manifest.csv"] = hashlib.sha256((args.source / "source_data/base_manifest.csv").read_bytes()).hexdigest()
    protocol = {
        "source": str(args.source.resolve()), "source_sha256": signatures,
        "split": "held-out test only; no model retraining or GPU inference",
        "deduplication": "one video per patient/lab timestamp, smallest lab-to-capture-interval distance, then midpoint distance, then video ID",
        "pairing": "adjacent distinct lab events; reject reversed/non-increasing video chronology; do not bridge excluded pairs",
        "direction": "exact observed rise/fall; numeric tolerance 1e-6 raw units; unchanged truth excluded from direction metrics only; predicted ties are errors",
        "bootstrap": {"unit": "patient", "iterations": args.bootstrap, "seed": 20261006},
        "interpretation": "Change between two existing predictions, not a retrained delta model. Labs may be offset from the video by up to 24h. Selected longest trajectories are descriptive only.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    plot(summary, pairs, events, figures)
    lines = ["# Native-224 / 24h longitudinal tracking", "", protocol["split"], "",
             "Repeated video matches to the same lab event are collapsed. Only increasing video/lab chronology is scored.",
             "Intervals use patient-cluster resampling. A zero-change prediction is the MAE baseline; negative skill is worse than this baseline.", "",
             "| Analyte | Patients | Pairs | Change r | Direction bACC | MAE skill | Within-patient r |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {ANALYTES[row['target']][0]} | {row['longitudinal_patients']} | {row['pairs']} | {row['delta_r']:.3f} | {row['direction_bacc']:.3f} | {row['mae_skill_vs_no_change']:.3f} | {row['within_patient_centered_r']:.3f} |")
    lines += ["", "Direction bACC applies only to nonzero changes and requires both directions. No clinical minimum-change threshold is imposed.",
              "Correlation alone does not establish calibrated tracking; read it with delta error and persistence skill.",
              "These comparisons concern the matched lab timestamps, not simultaneous blood draws or a causal recovery assessment."]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
