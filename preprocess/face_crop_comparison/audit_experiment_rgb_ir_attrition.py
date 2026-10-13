"""Read-only RGB+IR20 eligibility attrition for current Exp2/Exp6 task records."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_comparison import sha256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIRED = HERE / "outputs/rgb_ir_paired224"
OUT = PAIRED / "experiment_attrition"
EXPERIMENTS = {
    "exp2": ROOT / "study/exp2_face_pretrained_head32_regression/outputs/20frame_face224",
    "exp6": ROOT / "study/exp6_face_pair_lab_delta/outputs/face224",
}
LABELS = {"oxyhemoglobin_fraction": "Oxyhemoglobin fraction", "lactate_high": "Lactate",
          "urea_high": "Urea", "total_bilirubin_high": "Total bilirubin", "platelet_count_low": "Platelets",
          "hemoglobin_low": "Hemoglobin", "aa_po2_ratio_low": "A/a PO2 ratio", "creatinine_high": "Creatinine",
          "hematocrit_low": "Hematocrit", "egfr_low": "eGFR", "troponin_high": "Troponin I"}


def video_set(records, experiment):
    if experiment == "exp2":
        return set(records.video_id)
    return set(records.first_video_id) | set(records.second_video_id)


def event_set(records, experiment):
    if experiment == "exp2":
        return set(records.clinical_event_id)
    first = set(zip(records.hospital_id, records.first_lab_time_unix))
    return first | set(zip(records.hospital_id, records.second_lab_time_unix))


def filter_records(records, experiment, eligible):
    if experiment == "exp2":
        return records.video_id.isin(eligible)
    return records.first_video_id.isin(eligible) & records.second_video_id.isin(eligible)


def summary_row(before, after, experiment, target, split, eligible):
    row = {"experiment": experiment, "target": target, "split": split,
           "sample_unit": "video-label" if experiment == "exp2" else "chronological lab/video-pair delta",
           "samples_rgb": len(before), "samples_rgb_ir": len(after),
           "unique_videos_rgb": len(video_set(before, experiment)), "unique_videos_rgb_ir": len(video_set(after, experiment)),
           "patients_rgb": before.hospital_id.nunique(), "patients_rgb_ir": after.hospital_id.nunique(),
           "unique_lab_events_rgb": len(event_set(before, experiment)),
           "unique_lab_events_rgb_ir": len(event_set(after, experiment)),
           "directly_eligible_endpoint_videos": len(video_set(before, experiment) & eligible)}
    for name in ("samples", "unique_videos", "patients", "unique_lab_events"):
        row[f"lost_{name}"] = row[f"{name}_rgb"]-row[f"{name}_rgb_ir"]
        row[f"lost_{name}_fraction"] = row[f"lost_{name}"]/max(row[f"{name}_rgb"], 1)
    row["eligible_videos_orphaned_by_partner_loss"] = row["directly_eligible_endpoint_videos"]-row["unique_videos_rgb_ir"]
    if split in ("train", "val", "test"):
        views = 5 if split == "train" else 1
        row["frame_or_temporal_pair_inputs_rgb"] = len(before)*20*views
        row["frame_or_temporal_pair_inputs_rgb_ir"] = len(after)*20*views
    return row


def failure_reason(row):
    if row["pairs20_eligible"]:
        return "eligible20"
    if row["ir_status"] == "invalid_source":
        return "IR_packet_timestamp_index_invalid"
    if row["ir_status"] == "missing_or_empty_video":
        return "IR_source_missing_or_empty"
    if row["ir_status"] == "no_valid_frames":
        return "IR_no_quality_accepted_frames"
    if row["ir_status"] == "failed":
        return "IR_processing_failed"
    if not row["candidate_pairs"]:
        return "no_frozen_mutual_nearest_candidate"
    if row["retained_pairs"] < 20:
        return "fewer_than20_quality_retained_pairs"
    return "insufficient_nonadjacent_capacity_in_both_modalities"


def main():
    paired_summary = json.loads((PAIRED / "summary.json").read_text())
    if not paired_summary["complete_for_candidate_population"] or paired_summary["quality_unknown_candidate_pairs"]:
        raise RuntimeError("Paired IR quality audit is incomplete")
    videos = pd.read_csv(PAIRED / "tables/video_pair_counts.csv")
    if videos.video_id.duplicated().any():
        raise ValueError("Duplicate paired video IDs")
    eligible = set(videos.loc[videos.pairs20_eligible, "video_id"])
    selections = pd.read_csv(PAIRED / "tables/selected20_pairs.csv")
    if set(selections.video_id) != eligible or not selections.groupby("video_id").size().eq(20).all():
        raise ValueError("20-pair selections and eligible-video mask differ")
    for _, group in selections.groupby("video_id"):
        if (np.min(np.diff(group.rgb_source_frame_index)) < 2 or np.min(np.diff(group.ir_source_frame_index)) < 2
                or group.ir_source_frame_index.duplicated().any() or group.pair_delta_ms.gt(1000/30).any()):
            raise ValueError("Invalid paired selection")
    videos["exclusion_reason"] = videos.apply(failure_reason, axis=1)
    lookup = videos.set_index("video_id")
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    counts, union_rows, losses, source_hashes = [], [], [], {}
    dropped_video_rows, union_excluded_videos = [], []
    for experiment, directory in EXPERIMENTS.items():
        if experiment == "exp2":
            main_protocol = json.loads((directory / "main_protocol.json").read_text())
            extension = json.loads((directory / "direct_field_extension_protocol.json").read_text())
            targets = extension["targets"]
            expected_hashes = {**main_protocol["source_records_sha256"], **extension["record_hashes"][str(directory)]}
            lab_sha = main_protocol["lab_table_sha256"]
        else:
            manifest = json.loads((directory / "experiment_manifest.json").read_text())
            targets = manifest["targets"]
            expected_hashes = manifest["records_sha256"]
            lab_sha = manifest["lab_table_sha256"]
        before_parts, after_parts = [], []
        for target in targets:
            path = directory / f"task_records/{target}.csv"
            source_hashes[str(path)] = sha256(path)
            if source_hashes[str(path)] != expected_hashes[target]:
                raise RuntimeError(f"Task records changed since completed experiment: {path}")
            records = pd.read_csv(path, dtype={"hospital_id": str}, float_precision="round_trip")
            if records.groupby("hospital_id").split.nunique().gt(1).any():
                raise ValueError("Patient leakage in the reference task records")
            key = "video_id" if experiment == "exp2" else "pair_id"
            if records[key].duplicated().any():
                raise ValueError("Duplicate training sample IDs")
            if not video_set(records, experiment).issubset(lookup.index):
                raise ValueError("Reference includes videos not covered by the paired audit")
            if experiment == "exp6" and not np.allclose(records.raw_delta, records.second_value-records.first_value, rtol=0, atol=1e-9):
                raise ValueError("Reference raw deltas are inconsistent")
            keep = filter_records(records, experiment, eligible)
            after = records.loc[keep]
            for split in ("all", "train", "val", "test"):
                old = records if split == "all" else records.loc[records.split.eq(split)]
                new = after if split == "all" else after.loc[after.split.eq(split)]
                counts.append(summary_row(old, new, experiment, target, split, eligible))
            removed = records.loc[~keep].copy()
            removed.insert(0, "experiment", experiment)
            removed["audit_target"] = target
            if experiment == "exp2":
                removed["exclusion_reason"] = removed.video_id.map(lookup.exclusion_reason)
            else:
                removed["first_reason"] = removed.first_video_id.map(lookup.exclusion_reason)
                removed["second_reason"] = removed.second_video_id.map(lookup.exclusion_reason)
                removed["excluded_endpoint_pattern"] = np.where(
                    removed.first_video_id.isin(eligible), "second_only",
                    np.where(removed.second_video_id.isin(eligible), "first_only", "both"))
            losses.append(removed)
            before_parts.append(records.assign(audit_target=target))
            after_parts.append(after.assign(audit_target=target))
            for video_id in sorted(video_set(records, experiment)-video_set(after, experiment)):
                dropped_video_rows.append({"experiment": experiment, "target": target, "video_id": video_id,
                    "paired20_eligible": video_id in eligible, "exclusion_reason": lookup.loc[video_id, "exclusion_reason"],
                    "orphaned_by_partner": video_id in eligible})
            print(f"[attrition] {experiment}/{target}: {len(records)} -> {len(after)}", flush=True)
        before, after = pd.concat(before_parts, ignore_index=True), pd.concat(after_parts, ignore_index=True)
        old_videos, new_videos = video_set(before, experiment), video_set(after, experiment)
        for video_id in sorted(old_videos-new_videos):
            union_excluded_videos.append({"experiment": experiment, "video_id": video_id,
                "exclusion_reason": lookup.loc[video_id, "exclusion_reason"],
                "orphaned_by_partner": video_id in eligible})
        if experiment == "exp2":
            old_units, new_units = before.video_id.nunique(), after.video_id.nunique()
        else:
            old_units = len(before[["first_video_id", "second_video_id"]].drop_duplicates())
            new_units = len(after[["first_video_id", "second_video_id"]].drop_duplicates())
        union_rows.append({"experiment": experiment, "targets": len(targets), "lab_table_sha256": lab_sha,
            "unique_video_or_video_pair_units_rgb": old_units, "unique_video_or_video_pair_units_rgb_ir": new_units,
            "lost_unique_units": old_units-new_units, "lost_unique_units_fraction": (old_units-new_units)/old_units,
            "target_specific_samples_rgb": len(before), "target_specific_samples_rgb_ir": len(after),
            "patients_rgb": before.hospital_id.nunique(), "patients_rgb_ir": after.hospital_id.nunique(),
            "unique_videos_rgb": len(old_videos), "unique_videos_rgb_ir": len(new_videos),
            "directly_eligible_videos": len(old_videos & eligible),
            "eligible_videos_orphaned_by_partner_loss": len(old_videos & eligible)-len(new_videos),
            "all_retained_rgb_source_frames_in_reference_videos": int(lookup.loc[sorted(old_videos), "rgb_frames"].sum()),
            "quality_retained_rgb_ir_pairs_in_surviving_videos": int(lookup.loc[sorted(new_videos), "retained_pairs"].sum())})
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / "tables/target_and_split_counts.csv", index=False)
    union = pd.DataFrame(union_rows)
    union.to_csv(OUT / "tables/experiment_union_counts.csv", index=False)
    pd.DataFrame(union_excluded_videos).to_csv(OUT / "tables/experiment_union_excluded_videos.csv", index=False)
    pd.concat(losses, ignore_index=True).to_csv(OUT / "tables/excluded_task_samples.csv", index=False)
    dropped = pd.DataFrame(dropped_video_rows)
    dropped.to_csv(OUT / "tables/excluded_task_videos.csv", index=False)
    dropped.groupby(["experiment", "target", "exclusion_reason", "orphaned_by_partner"], as_index=False).size().to_csv(OUT / "tables/video_exclusion_reasons.csv", index=False)
    all_counts = counts.loc[counts.split.eq("all")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, experiment in zip(axes, EXPERIMENTS):
        rows = all_counts.loc[all_counts.experiment.eq(experiment)]
        y = np.arange(len(rows))
        ax.barh(y, rows.samples_rgb, color="#d7dde0", label="RGB reference")
        ax.barh(y, rows.samples_rgb_ir, color="#297e69", label="RGB+IR20 eligible")
        ax.set_yticks(y, [LABELS[t] for t in rows.target])
        ax.invert_yaxis()
        ax.set_xlabel("Video-label samples" if experiment == "exp2" else "Lab/video-pair delta samples")
        ax.set_title("Single-time face regression" if experiment == "exp2" else "Two-time delta regression")
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, row in enumerate(rows.itertuples()):
            ax.text(row.samples_rgb+5, i, f"-{row.lost_samples} ({row.lost_samples_fraction:.1%})", va="center", fontsize=8)
        ax.set_xlim(0, rows.samples_rgb.max()*1.27)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "figures/experiment_sample_retention.png", dpi=180)
    plt.close(fig)
    manifest = {"generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "experiment_roots": {k: str(v) for k, v in EXPERIMENTS.items()},
        "reference_record_hashes": source_hashes, "paired_summary_sha256": sha256(PAIRED / "summary.json"),
        "paired_video_table_sha256": sha256(PAIRED / "tables/video_pair_counts.csv"),
        "policy": "filter existing samples by joint20 eligibility, Exp6 requires both endpoints; no rematching/bridging, relabeling, scaling or re-splitting",
        "frame_policy": "twenty joint frames may be reselected; not restricted to the old sampled twenty RGB frames",
        "exp2_sample_unit": "one video/current-analyte label", "exp6_sample_unit": "one recorded two-time analyte delta",
        "cross_target_totals": "target-specific sample totals repeat video IDs/patients; unique union totals are reported separately",
        "experiments_modified_or_retrained": False}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    lines = ["# RGB+IR20 Attrition In Current Face Regression Experiments", "",
        "Read-only audit of completed native224 primary Exp2 (10 targets) and Exp6 (11 targets, additionally troponin). Old outputs/20frame and legacy Exp6 root results are not used.", "",
        "Use the frozen mutual-nearest <=33.333 ms population, IR quality intersection, unique raw-frame gaps >=2 in both modalities, and successfully decoded 20-pair selections. Every original label, patient split and Exp6 chronological pair is unchanged. Lost endpoints are not bridged to create new delta samples.", "",
        "Twenty joint frames can be reselected per surviving video. This is dataset eligibility loss, not a check that the old sampled twenty RGB images each have IR. A surviving sample still supplies twenty frame positions; five views expand training only, not validation/test.", "",
        "## Unique Union", "", "|Experiment|Unique sample units RGB -> RGB+IR|Patients RGB -> RGB+IR|Participating videos RGB -> RGB+IR|",
        "|---|---:|---:|---:|"]
    for row in union.itertuples(index=False):
        lines.append(f"|{row.experiment}|{row.unique_video_or_video_pair_units_rgb} -> {row.unique_video_or_video_pair_units_rgb_ir}|{row.patients_rgb} -> {row.patients_rgb_ir}|{row.unique_videos_rgb} -> {row.unique_videos_rgb_ir}|")
    for experiment in EXPERIMENTS:
        lines += ["", f"## {experiment.upper()} Per Target", "", "|Analyte|Samples RGB|Samples RGB+IR|Lost|Lost %|Patients RGB -> RGB+IR|Unique labs RGB -> RGB+IR|",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in all_counts.loc[all_counts.experiment.eq(experiment)].itertuples(index=False):
            lines.append(f"|{LABELS[row.target]}|{row.samples_rgb}|{row.samples_rgb_ir}|{row.lost_samples}|{row.lost_samples_fraction:.2%}|{row.patients_rgb} -> {row.patients_rgb_ir}|{row.unique_lab_events_rgb} -> {row.unique_lab_events_rgb_ir}|")
    lines += ["", "## Interpretation", "",
        "Exp2 counts video-labels. Exp6 counts two-time lab/video pairs, so loss of either video drops the complete delta sample; some individually eligible endpoint videos become orphaned and cannot participate in any retained original pair.", "",
        "Patient loss means losing every usable sample for that target, not merely losing one video. Counts cannot be summed across targets to obtain independent patients or unique videos. Unique union counts remove this duplication; report-time lab events and sample/view counts are distinct units.", "",
        "The full source-frame counts in the union CSV are descriptive storage/sample-pool sizes, not training input counts for a twenty-frame experiment. Detailed train/val/test counts and per-epoch view inputs are in tables/target_and_split_counts.csv; excluded sample IDs and reasons are retained separately."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(union.to_string(index=False), flush=True)
    print(f"Report: {OUT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
