"""Patient-weighted lab changes in continuous +/-6,12,24h video windows."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.plot_layout import target_grid_shape, target_grid_figsize
from study.exp2_face_pretrained_head32_regression.config import TARGETS, SCORE_DEFINITIONS
from study.exp2_face_pretrained_head32_regression.plot_results import TASK_LABELS, TASK_UNITS
from study.exp2_face_pretrained_head32_regression.data import validate_source_data


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = STUDY / "exp2_face_pretrained_head32_regression/outputs/20frame_face224"
PROTOCOL = STUDY / "common/outputs/face_main_24h_frame_loss/protocol.json"
HOURS = (6, 12, 24)
COLORS = ("#379A86", "#2878B5", "#CB6547")
BOOTSTRAPS = 2000
SEED = 20261008


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_statistics(times, values, start, end, admission, discharge, hours, threshold, direction, iqr):
    lower = max(start - hours * 3600, admission)
    upper = min(end + hours * 3600, discharge)
    first = np.searchsorted(times, lower, side="left")
    last = np.searchsorted(times, upper, side="right")
    times, values = times[first:last], values[first:last]
    count = len(values)
    result = {
        "window_start_unix": lower, "window_end_unix": upper,
        "window_clipped_by_admission": int(admission > start - hours * 3600),
        "window_clipped_by_discharge": int(discharge < end + hours * 3600),
        "n_lab_events": count, "has_change_measurement": int(count >= 2),
        "brackets_video_interval": int(count >= 2 and times[0] <= start and times[-1] >= end),
    }
    metrics = (
        "first_value", "last_value", "first_lab_time_unix", "last_lab_time_unix",
        "observed_span_h", "signed_change", "absolute_change", "value_range",
        "sample_sd", "within_window_iqr", "signed_change_train_iqr",
        "range_train_iqr", "sd_train_iqr", "threshold_crossing",
        "nearest_value", "max_deviation_from_nearest", "nearest_match_delta_h",
    )
    result.update({name: np.nan for name in metrics})
    if not count:
        return result
    distance = np.maximum(np.maximum(start - times, times - end), 0)
    midpoint_distance = np.abs(times - (start + end) / 2)
    nearest = np.lexsort((times, midpoint_distance, distance))[0]
    result.update(nearest_value=float(values[nearest]), nearest_match_delta_h=float(distance[nearest] / 3600))
    if count < 2:
        return result
    change = float(values[-1] - values[0])
    value_range = float(np.ptp(values))
    sd = float(values.std(ddof=1))
    abnormal = values < threshold if direction == "low" else values > threshold
    result.update(
        first_value=float(values[0]), last_value=float(values[-1]),
        first_lab_time_unix=float(times[0]), last_lab_time_unix=float(times[-1]),
        observed_span_h=float((times[-1] - times[0]) / 3600),
        signed_change=change, absolute_change=abs(change), value_range=value_range,
        sample_sd=sd, within_window_iqr=float(np.quantile(values, .75) - np.quantile(values, .25)),
        signed_change_train_iqr=change / iqr, range_train_iqr=value_range / iqr, sd_train_iqr=sd / iqr,
        threshold_crossing=int(abnormal.any() and (~abnormal).any()),
        max_deviation_from_nearest=float(np.max(np.abs(values - values[nearest]))),
    )
    return result


def load_inputs():
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(STUDY.parent / "merged_lab_tests.csv") != protocol["lab_table_sha256"]:
        raise RuntimeError("The current matched source does not use the latest lab table")
    quality = validate_source_data(SOURCE / "source_data", 24)
    base = pd.read_csv(SOURCE / "source_data/base_manifest.csv", dtype={"hospital_id": str, "video_id": str})
    task_tables = []
    for target in TARGETS:
        path = SOURCE / f"task_records/{target}.csv"
        if sha256(path) != protocol["source_records_sha256"][target]:
            raise RuntimeError(f"Task cohort changed: {target}")
        task_tables.append(pd.read_csv(path, dtype={"hospital_id": str, "video_id": str}))
    eligible = pd.concat(task_tables, ignore_index=True)[["video_id", "hospital_id"]].drop_duplicates()
    if eligible.video_id.duplicated().any():
        raise RuntimeError("Video hospital identity is inconsistent across targets")
    videos = eligible.merge(base, on=["video_id", "hospital_id"], how="left", validate="one_to_one")
    required = ["capture_start_unix", "capture_end_unix", "admission_unix", "discharge_unix"]
    if videos[required].isna().any().any():
        raise RuntimeError("Missing clinical capture/hospitalization bounds")
    lab_path = SOURCE / "source_data/lab_timeseries.csv"
    if sha256(lab_path) != quality["source_fingerprints"]["lab_timeseries_cache"]["sha256"]:
        raise RuntimeError("Canonical lab time series changed")
    labs = pd.read_csv(lab_path, dtype={"hospital_id": str})
    if labs.duplicated(["hospital_id", "analyte", "timestamp_unix"]).any():
        raise RuntimeError("Lab event duplicates were not collapsed")
    if not np.isfinite(labs[["value", "timestamp_unix"]]).all().all():
        raise RuntimeError("Non-finite canonical laboratory values/times")
    scalers = json.loads((SOURCE / "target_scalers.json").read_text())["targets"]
    return videos, labs, scalers, quality, protocol


def analyze_windows(videos, labs, scalers):
    lookups = {}
    for (patient, analyte), group in labs.groupby(["hospital_id", "analyte"]):
        group = group.sort_values("timestamp_unix")
        lookups[patient, analyte] = (group.timestamp_unix.to_numpy(float), group.value.to_numpy(float))
    rows = []
    for target in TARGETS:
        definition = SCORE_DEFINITIONS[target]
        analyte = definition["value_column"].removesuffix("_value")
        for video in videos.itertuples(index=False):
            times, values = lookups.get((video.hospital_id, analyte), (np.array([]), np.array([])))
            threshold = definition["threshold"]
            if isinstance(threshold, dict):
                threshold = threshold["male"] if video.sex == "男" else threshold["other"]
            for hours in HOURS:
                row = {"target": target, "hospital_id": video.hospital_id, "video_id": video.video_id,
                       "window_half_width_h": hours, "capture_start_unix": video.capture_start_unix,
                       "capture_end_unix": video.capture_end_unix, "clinical_threshold": threshold,
                       "training_iqr": scalers[target]["iqr"]}
                row.update(window_statistics(times, values, video.capture_start_unix, video.capture_end_unix,
                                             video.admission_unix, video.discharge_unix, hours,
                                             threshold, definition["direction"], scalers[target]["iqr"]))
                rows.append(row)
    return pd.DataFrame(rows)


def patient_statistics(windows, cohort):
    variables = ("signed_change", "absolute_change", "value_range", "sample_sd", "within_window_iqr",
                 "signed_change_train_iqr", "range_train_iqr", "sd_train_iqr", "observed_span_h", "n_lab_events",
                 "max_deviation_from_nearest")
    aggregations = {name: (name, "median") for name in variables}
    aggregations.update(videos=("video_id", "size"), threshold_crossing=("threshold_crossing", "mean"),
                        brackets_video_fraction=("brackets_video_interval", "mean"))
    result = windows.groupby(["target", "window_half_width_h", "hospital_id"]).agg(**aggregations).reset_index()
    result.insert(0, "cohort", cohort)
    return result


def bootstrap_interval(values, statistic, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))]
    estimate = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return tuple(np.quantile(estimate, [.025, .975]))


def summarize(windows):
    eligible = windows.loc[windows.has_change_measurement.eq(1)].copy()
    keys = eligible.loc[eligible.window_half_width_h.eq(6), ["target", "video_id"]]
    common = eligible.merge(keys, on=["target", "video_id"], validate="many_to_one")
    patient_tables = [patient_statistics(eligible, "all_eligible"), patient_statistics(common, "common_6h_videos")]
    bracketed = eligible.loc[eligible.brackets_video_interval.eq(1)]
    patient_tables.append(patient_statistics(bracketed, "bracketing_video"))
    patients = pd.concat(patient_tables, ignore_index=True)
    metrics = ("signed_change", "absolute_change", "value_range", "sample_sd", "observed_span_h", "n_lab_events",
               "signed_change_train_iqr", "range_train_iqr", "sd_train_iqr", "max_deviation_from_nearest", "threshold_crossing")
    rng = np.random.default_rng(SEED)
    summaries = []
    for (cohort, target, hours), group in patients.groupby(["cohort", "target", "window_half_width_h"]):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            statistic = "mean" if metric == "threshold_crossing" else "median"
            low, high = bootstrap_interval(values, statistic, rng)
            summaries.append({"cohort": cohort, "target": target, "window_half_width_h": hours, "metric": metric,
                              "patients": len(group), "videos": int(group.videos.sum()),
                              "patient_level_statistic": statistic, "estimate": float(values.mean() if statistic == "mean" else np.median(values)),
                              "q25": float(np.quantile(values,.25)), "q75": float(np.quantile(values,.75)),
                              "p05": float(np.quantile(values,.05)), "p95": float(np.quantile(values,.95)),
                              "minimum": float(values.min()), "maximum": float(values.max()),
                              "ci95_low": low, "ci95_high": high})
    coverage = []
    for (target, hours), group in windows.groupby(["target", "window_half_width_h"]):
        multi = group.loc[group.has_change_measurement.eq(1)]
        coverage.append({"target": target, "window_half_width_h": hours, "candidate_videos": len(group),
                         "candidate_patients": group.hospital_id.nunique(), "zero_test_videos": int(group.n_lab_events.eq(0).sum()),
                         "single_test_videos": int(group.n_lab_events.eq(1).sum()), "multi_test_videos": len(multi),
                         "multi_test_patients": multi.hospital_id.nunique(), "multi_test_fraction": len(multi)/len(group),
                         "bracketing_video_count": int(multi.brackets_video_interval.sum())})
    return patients, pd.DataFrame(summaries), pd.DataFrame(coverage)


def panels():
    rows, columns = target_grid_shape(len(TARGETS))
    return plt.subplots(rows, columns, figsize=target_grid_figsize(rows, columns), squeeze=False)


def save(figure, figures, name):
    figure.tight_layout(rect=(0,.05,1,.95))
    for extension in ("png", "pdf"):
        figure.savefig(figures/f"{name}.{extension}", dpi=200)
    plt.close(figure)


def plot_all(patients, summary, coverage, output):
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    figure, axes = panels()
    for axis, target in zip(axes.flat, TARGETS):
        selected = coverage.loc[coverage.target.eq(target)].set_index("window_half_width_h").loc[list(HOURS)]
        base = np.zeros(3)
        for column, label, color in (("zero_test_videos","No test","#E3E6EA"),
                                     ("single_test_videos","One test","#9DAAB7"),
                                     ("multi_test_videos",">=2 tests","#2878B5")):
            height = selected[column].to_numpy()/selected.candidate_videos.to_numpy()*100
            axis.bar(range(3), height, bottom=base, color=color, label=label)
            base += height
        labels = [f"+/-{h}h\nV={int(selected.loc[h,'multi_test_videos'])}, P={int(selected.loc[h,'multi_test_patients'])}" for h in HOURS]
        axis.set(title=TASK_LABELS[target], xticks=range(3), xticklabels=labels, ylabel="Video windows (%)", ylim=(0,105))
        axis.tick_params(axis="x",labelsize=8);axis.grid(axis="y",alpha=.15);axis.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc="lower center",ncol=3)
    figure.suptitle("Independent lab-event coverage around the same native224 video cohort")
    save(figure,figures,"window_measurement_coverage")
    paired = patients.loc[patients.cohort.eq("common_6h_videos")]
    for metric, name, heading in (
        ("value_range","window_range_common_cohort","Within-window maximum minus minimum"),
        ("signed_change","window_net_change_common_cohort","Last minus first laboratory value"),
        ("sample_sd","window_sd_common_cohort","Within-window laboratory SD"),
    ):
        figure,axes=panels()
        for axis,target in zip(axes.flat,TARGETS):
            selected=paired.loc[paired.target.eq(target)]
            data=[selected.loc[selected.window_half_width_h.eq(h),metric].to_numpy() for h in HOURS]
            if not all(len(values) for values in data):
                axis.text(.5,.5,"No paired repeated-measurement windows",ha="center",va="center",transform=axis.transAxes)
                axis.set(title=TASK_LABELS[target]);continue
            boxes=axis.boxplot(data,positions=range(3),widths=.6,patch_artist=True,showfliers=False,whis=(5,95),medianprops={"color":"#222222"})
            for box,color in zip(boxes["boxes"],COLORS):box.set_facecolor(color);box.set_alpha(.7)
            counts=selected.loc[selected.window_half_width_h.eq(6)]
            unit="percentage points" if TASK_UNITS[target]=="%" else TASK_UNITS[target]
            axis.set(title=f"{TASK_LABELS[target]}\nV={int(counts.videos.sum())}, P={len(counts)}",xticks=range(3),
                     xticklabels=[f"+/-{h}h" for h in HOURS],ylabel=unit)
            if metric=="signed_change":axis.axhline(0,color="#666666",linewidth=.7,linestyle="--")
            axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True)
        figure.suptitle(f"{heading} | same videos across all three windows")
        figure.text(.5,.01,"One patient median across their eligible videos. Boxes: IQR; center: median; whiskers: 5th-95th percentiles. Full extremes remain in CSV.",ha="center",fontsize=8)
        save(figure,figures,name)
    for metric,name,heading,xlabel,multiplier in (
        ("range_train_iqr","window_range_iqr_scaled","Patient-weighted variability across analytes","Range / training-set IQR",1),
        ("threshold_crossing","window_threshold_crossing","Windows containing both normal and abnormal results","Patient-mean fraction of mixed-label windows (%)",100),
    ):
        figure,axis=plt.subplots(figsize=(11,7))
        y=np.arange(len(TARGETS));height=.24
        for position,(hours,color) in enumerate(zip(HOURS,COLORS)):
            table=summary.loc[summary.cohort.eq("common_6h_videos") & summary.metric.eq(metric) & summary.window_half_width_h.eq(hours)].set_index("target").reindex(TARGETS)
            value=table.estimate.to_numpy()*multiplier
            axis.barh(y+(position-1)*height,value,height,color=color,label=f"+/-{hours}h")
            low=table.ci95_low.to_numpy()*multiplier;high=table.ci95_high.to_numpy()*multiplier
            good=np.isfinite(value)&np.isfinite(low)&np.isfinite(high)
            axis.errorbar(value[good],(y+(position-1)*height)[good],xerr=np.maximum(np.stack([value[good]-low[good],high[good]-value[good]]),0),fmt="none",ecolor="#333333",capsize=2,linewidth=.8)
        axis.set(yticks=y,yticklabels=[TASK_LABELS[target] for target in TARGETS],xlabel=xlabel,title=heading)
        axis.invert_yaxis();axis.legend();axis.grid(axis="x",alpha=.2);axis.set_axisbelow(True)
        figure.text(.5,.01,"Same repeated-measurement videos across windows; each patient has equal weight. Error bars: 95% patient-bootstrap CI.",ha="center",fontsize=8)
        save(figure,figures,name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only",action="store_true")
    args=parser.parse_args()
    output=HERE/"outputs"
    output.mkdir(parents=True,exist_ok=True)
    if args.plots_only:
        plot_all(pd.read_csv(output/"patient_statistics.csv"),pd.read_csv(output/"variability_summary.csv"),pd.read_csv(output/"window_coverage.csv"),output)
        return
    videos,labs,scalers,quality,protocol=load_inputs()
    windows=analyze_windows(videos,labs,scalers)
    patients,summary,coverage=summarize(windows)
    windows.to_csv(output/"video_window_statistics.csv",index=False)
    patients.to_csv(output/"patient_statistics.csv",index=False)
    summary.to_csv(output/"variability_summary.csv",index=False)
    coverage.to_csv(output/"window_coverage.csv",index=False)
    main_table=summary.loc[summary.cohort.eq("all_eligible") & summary.metric.eq("value_range")].pivot(index="target",columns="window_half_width_h",values="estimate").reindex(TARGETS)
    main_table.columns=[f"median_patient_range_{h}h" for h in main_table.columns]
    main_table.to_csv(output/"patient_weighted_range_table.csv")
    manifest={
        "lab_table_sha256":protocol["lab_table_sha256"],"source":str(SOURCE),
        "base_manifest_sha256":sha256(SOURCE/"source_data/base_manifest.csv"),
        "lab_timeseries_sha256":sha256(SOURCE/"source_data/lab_timeseries.csv"),
        "scalers_sha256":sha256(SOURCE/"target_scalers.json"),
        "video_cohort":"union of native224/20frame main task records, not limited to a complete eight-analyte panel",
        "videos":len(videos),"patients":int(videos.hospital_id.nunique()),"targets":list(TARGETS),"windows_h":list(HOURS),
        "window_definition":"[original video start - H, original video end + H], inclusive, intersected with same admission",
        "time_basis":"validated Session Timestamp and original source interval; Asia/Shanghai lab-report times stored as Unix seconds",
        "lab_source_policies":quality["analyte_source_policies"],
        "lab_event_unit":"one canonical hospital_id/analyte/report timestamp; source duplicates already collapsed",
        "insufficient_windows":"0 or 1 event: change/range/SD/threshold-crossing are missing, never zero-imputed",
        "cohorts":{"all_eligible":"each H's windows with at least two events", "common_6h_videos":"same target-video pairs with at least two events at +/-6h in all windows", "bracketing_video":"at least one lab before video start and one after video end"},
        "patient_weighting":"one per-patient median of eligible window magnitudes; threshold crossing uses one per-patient mean proportion",
        "crossing_definition":"both normal and abnormal values among window events; not a change of the selected nearest label",
        "iqr_normalization":"divide magnitude by the existing model's train-only target IQR; no new scaling fit",
        "bootstrap":{"unit":"patient","resamples":BOOTSTRAPS,"seed":SEED,"ci":"percentile 2.5-97.5%"},
        "cautions":["overlapping video windows can reuse lab events; no independence claim between windows", "range and SD depend on event count and measured span", "paired figures exclude windows without repeated measurements at +/-6h"],
        "model_training_or_inference":False,
    }
    if sha256(STUDY.parent/"merged_lab_tests.csv")!=protocol["lab_table_sha256"]:raise RuntimeError("Lab table changed during analysis")
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    plot_all(patients,summary,coverage,output)
    print(main_table.to_string(),flush=True)
    print(f"[analysis-complete] videos={len(videos)} patients={videos.hospital_id.nunique()} output={output}",flush=True)


if __name__=="__main__":main()
