"""Audit frame-count admission without altering crops, matches or training."""

from collections import Counter
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.common.face_video import raw_root, resolve_video, scan_ffv1, decode_indexed_frame
from study.exp2_face_pretrained_head32_regression.config import DATA_ROOT, ALL_REGRESSION_TARGETS
from .run_analysis import _sha256, _figure_style
from .video24h_matching import HERE, REFERENCE, TABLES as MATCH_TABLES, nearest_sorted


OUTPUT = HERE / "outputs"
TABLES = MATCH_TABLES / "frame_budget"
THRESHOLDS = (1, 2, 3, 5, 8, 10, 12, 15, 16, 18, 19, 20)


def spaced_positions(indices, gap=2):
    indices = np.asarray(indices, np.int64)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Source frame IDs must be strictly increasing")
    selected, last = [], None
    for i, frame in enumerate(indices):
        if last is None or frame - last >= gap:
            selected.append(i)
            last = frame
    return np.asarray(selected, dtype=np.int64)


def uniform_positions(indices, count, gap=2):
    """Mirror current 5%-95% quantiles and full-span fallback for arbitrary N."""
    indices = np.asarray(indices, np.int64)
    if len(indices) < count:
        return None
    if count == 1:
        return np.asarray([int(round((len(indices) - 1) / 2))])
    quantiles = [.05 + .90 * i / (count - 1) for i in range(count)]
    chosen = np.rint(np.asarray(quantiles) * (len(indices) - 1)).astype(int)
    if len(set(chosen)) != count or np.min(np.diff(indices[chosen])) < gap:
        chosen = np.rint(np.linspace(0, len(indices) - 1, count)).astype(int)
    if len(set(chosen)) != count or np.min(np.diff(indices[chosen])) < gap:
        return None
    return chosen


def exclusion_category(reason):
    if reason == "cannot_select_20_nonadjacent_accepted_frames":
        return "帧数或20帧抽样限制"
    if reason == "no_valid_face224_frames":
        return "没有通过质量筛选的人脸帧"
    if reason == "missing_or_empty_original_rgb_video":
        return "原始RGB视频缺失或为空"
    if reason.startswith("invalid_original_frame_timestamp_mapping:"):
        return "原始视频与时间戳映射错误"
    return "其他结构或验证错误"


def inspect_videos(videos):
    rows, rejection_totals, rejection_videos = [], Counter(), Counter()
    for ordinal, video in enumerate(videos.itertuples(index=False), 1):
        row = video._asdict()
        directory = raw_root() / f"{video.mirror}_data/patient_{video.lab_patient_id:06d}"
        metadata_path = directory / "face224_metadata.json"
        row.update(valid_face_frames=0, source_frames=0, max_gap2_frames=0, validated_completed_crop=False,
                   retained_span_seconds=np.nan, confidence_min=np.nan, confidence_median=np.nan,
                   border_only_rejected_frames=0, low_confidence_only_rejected_frames=0,
                   twenty_failure_category="eligible" if video.native224_20frame_eligible else exclusion_category(video.frame_status))
        for count in THRESHOLDS:
            row[f"current_n{count}"] = row[f"gap1_n{count}"] = row[f"constrained_n{count}"] = False
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            row["source_frames"] = int(metadata.get("source_frames", 0))
            row["valid_face_frames"] = int(metadata.get("retained_frames", 0))
            if metadata["status"] == "no_valid_frames":
                rejections = metadata.get("rejection_counts", {})
                rejection_totals.update(rejections)
                rejection_videos.update({reason: 1 for reason, count in rejections.items() if count})
                row["border_only_rejected_frames"] = rejections.get("expanded_box_outside_source_over_10_percent", 0)
                row["low_confidence_only_rejected_frames"] = rejections.get("low_confidence", 0)
            if metadata["status"] == "completed":
                resolve_video(video, DATA_ROOT)
                mapping = pd.read_csv(directory / "face224_frames.csv")
                indices = mapping.source_frame_index.to_numpy(np.int64)
                if (len(mapping) != row["valid_face_frames"] or not np.array_equal(mapping.output_frame_index, np.arange(len(mapping)))
                        or np.any(np.diff(mapping.source_elapsed_seconds) <= 0) or np.any(np.diff(mapping.encoded_pts) <= 0)
                        or mapping.confidence.isna().any() or mapping.confidence.lt(.75).any()):
                    raise RuntimeError(f"Retained frame mapping/quality changed: {video.video_id}")
                expected = metadata["session_time_unix"] + mapping.source_elapsed_seconds.to_numpy()
                np.testing.assert_allclose(mapping.canonical_session_frame_time_unix, expected, rtol=0, atol=1e-6)
                spaced = spaced_positions(indices)
                row.update(max_gap2_frames=len(spaced), validated_completed_crop=True,
                           retained_span_seconds=float(mapping.source_elapsed_seconds.iloc[-1] - mapping.source_elapsed_seconds.iloc[0]),
                           confidence_min=float(mapping.confidence.min()), confidence_median=float(mapping.confidence.median()))
                for count in THRESHOLDS:
                    row[f"current_n{count}"] = uniform_positions(indices, count) is not None
                    row[f"gap1_n{count}"] = uniform_positions(indices, count, gap=1) is not None
                    row[f"constrained_n{count}"] = len(spaced) >= count
                if row["current_n20"] != video.native224_20frame_eligible:
                    raise RuntimeError(f"Current 20-frame eligibility changed: {video.video_id}")
                if not row["current_n20"]:
                    row["twenty_failure_category"] = ("有效帧总数不足20" if len(indices) < 20 else
                        "原始帧间隔≥2时最多不足20帧" if len(spaced) < 20 else "均匀抽样失败但可重选20帧")
        rows.append(row)
        if ordinal % 250 == 0:
            print(f"[frame-audit] {ordinal}/{len(videos)} sidecars; no decoding or crop changes", flush=True)
    result = pd.DataFrame(rows)
    if not np.array_equal(result.current_n20, videos.native224_20frame_eligible):
        raise RuntimeError("Audit did not exactly reproduce the existing 20-frame cohort")
    quality = pd.DataFrame([{"reason": reason, "rejected_frames": count, "affected_zero_retention_videos": rejection_videos[reason]}
                            for reason, count in rejection_totals.most_common()])
    return result, quality


def verify_recovered_twenty(frames):
    rows = []
    for video in frames.loc[~frames.current_n20 & frames.constrained_n20].itertuples(index=False):
        path = Path(resolve_video(video, DATA_ROOT))
        mapping = pd.read_csv(path.with_name("face224_frames.csv"))
        indices = mapping.source_frame_index.to_numpy(np.int64)
        spaced = spaced_positions(indices)
        positions = spaced[uniform_positions(indices[spaced], 20)]
        quantiles = positions / (len(indices) - 1)
        starts, ends, source_ids, extra, failures = scan_ffv1(path, "20frame", quantiles, 2)
        if failures or len(starts) != 20:
            raise RuntimeError(f"Proposed 20-frame selection failed: {video.video_id}")
        index = SimpleNamespace(starts=starts, ends=ends, video_formats=["ffv1"], codec_extradata=[extra])
        with path.open("rb") as handle:
            decoders = OrderedDict()
            for i in range(20):
                decoded = decode_indexed_frame(index, 0, i, handle, decoders)
                assert decoded.shape == (3, 224, 224)
        rows.append({"video_id": video.video_id, "hospital_id": video.hospital_id, "decoded_frames": 20,
                     "minimum_source_frame_gap": int(np.diff(source_ids).min()),
                     "minimum_selected_confidence": float(mapping.confidence.iloc[positions].min()),
                     "source_frame_indices": json.dumps(source_ids), "all_packets_decodable": True})
    return pd.DataFrame(rows)


def task_membership(videos):
    base = pd.read_csv(REFERENCE / "source_data/base_manifest.csv", dtype={"hospital_id": str, "video_id": str})
    memberships = {target: set(base.loc[base[target].notna(), "video_id"]) for target in ALL_REGRESSION_TARGETS[:8]}
    direct = pd.read_csv(REFERENCE / "source_data/direct_fields/lab_timeseries.csv", dtype={"hospital_id": str})
    names = {"hematocrit_low": "hematocrit", "egfr_low": "egfr_creatinine"}
    for target, analyte in names.items():
        table = direct.loc[direct.analyte.eq(analyte)].groupby(["hospital_id", "timestamp_unix"], as_index=False).value.median()
        lookup = {patient: group.sort_values("timestamp_unix") for patient, group in table.groupby("hospital_id")}
        found = set()
        for video in videos.itertuples(index=False):
            group = lookup.get(video.hospital_id)
            if group is not None and nearest_sorted(group.timestamp_unix.to_numpy(), group.value.to_numpy(), video) is not None:
                found.add(video.video_id)
        memberships[target] = found
    return memberships


def scenario_metrics(frame_table, scopes):
    rows = []
    for scope, ids in scopes.items():
        subset = frame_table.loc[frame_table.video_id.isin(ids)]
        baseline = subset.loc[subset.current_n20]
        old_ids, old_patients = set(baseline.video_id), set(baseline.hospital_id)
        for policy in ("current", "constrained", "gap1"):
            for count in THRESHOLDS:
                selected = subset.loc[subset[f"{policy}_n{count}"]]
                rows.append({"scope": scope, "policy": policy, "frames_per_video": count, "eligible_videos": len(selected),
                             "eligible_patients": selected.hospital_id.nunique(), "added_videos_vs_current20": len(set(selected.video_id) - old_ids),
                             "lost_videos_vs_current20": len(old_ids - set(selected.video_id)),
                             "added_patients_vs_current20": len(set(selected.hospital_id) - old_patients),
                             "selected_original_frames": len(selected) * count, "five_view_inputs": len(selected) * count * 5,
                             "all_retained_original_frames_in_admitted_videos": int(selected.valid_face_frames.sum()),
                             "all_gap2_frames_in_admitted_videos": int(selected.max_gap2_frames.sum())})
        for minimum in (1, 2, 3, 5, 8, 10, 12, 15, 20):
            selected = subset.loc[subset.validated_completed_crop & subset.max_gap2_frames.ge(minimum)]
            adaptive_counts = np.minimum(selected.max_gap2_frames.to_numpy(), 20)
            rows.append({"scope": scope, "policy": "adaptive_cap20", "frames_per_video": minimum,
                         "eligible_videos": len(selected), "eligible_patients": selected.hospital_id.nunique(),
                         "added_videos_vs_current20": len(set(selected.video_id) - old_ids), "lost_videos_vs_current20": len(old_ids - set(selected.video_id)),
                         "added_patients_vs_current20": len(set(selected.hospital_id) - old_patients),
                         "selected_original_frames": int(adaptive_counts.sum()), "five_view_inputs": int(adaptive_counts.sum() * 5),
                         "all_retained_original_frames_in_admitted_videos": int(selected.valid_face_frames.sum()),
                         "all_gap2_frames_in_admitted_videos": int(selected.max_gap2_frames.sum())})
    return pd.DataFrame(rows)


def publish(frames, summary, quality, manifests, recovered):
    TABLES.mkdir(parents=True, exist_ok=True)
    frames.to_csv(TABLES / "per_video_frame_counts.csv", index=False)
    summary.to_csv(TABLES / "frame_budget_scenarios.csv", index=False)
    quality.to_csv(TABLES / "zero_retention_rejections.csv", index=False)
    recovered.to_csv(TABLES / "recovered20_packet_verification.csv", index=False)
    zero = frames.loc[frames.twenty_failure_category.eq("没有通过质量筛选的人脸帧")]
    zero[["video_id", "hospital_id", "source_frames", "border_only_rejected_frames", "low_confidence_only_rejected_frames"]].to_csv(
        TABLES / "zero_retention_review_candidates.csv", index=False)
    failures = frames.loc[~frames.current_n20].groupby("twenty_failure_category").agg(videos=("video_id", "size"), patients=("hospital_id", "nunique")).reset_index()
    failures.to_csv(TABLES / "twenty_frame_exclusions.csv", index=False)
    short = frames.loc[frames.validated_completed_crop & ~frames.current_n20]
    short.valid_face_frames.value_counts().sort_index().rename_axis("valid_frames").reset_index(name="videos").to_csv(TABLES / "short_video_distribution.csv", index=False)
    _figure_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, scope, title in zip(axes[0], ("all_valid_sessions", "any_exact_lab_24h"), ("All validated sessions", "At least one exact 24h lab match")):
        for policy, label, color in (("current", "Current uniform / gap>=2", "#2878B5"),
                                     ("constrained", "Constrained uniform / gap>=2", "#278245"), ("gap1", "Allow adjacent frames", "#CB6547")):
            part = summary.loc[summary.scope.eq(scope) & summary.policy.eq(policy)].sort_values("frames_per_video")
            axis.plot(part.frames_per_video, part.eligible_videos, marker="o", label=label, color=color)
        axis.set(title=title, xlabel="Required frames per video", ylabel="Eligible videos"); axis.legend(fontsize=8); axis.grid(alpha=.2)
    part = summary.loc[summary.scope.eq("any_exact_lab_24h") & summary.policy.eq("current")].sort_values("frames_per_video")
    axes[1, 0].plot(part.frames_per_video, part.selected_original_frames, marker="o", color="#2878B5", label="Fixed N per video")
    adaptive = summary.loc[summary.scope.eq("any_exact_lab_24h") & summary.policy.eq("adaptive_cap20")].sort_values("frames_per_video")
    axes[1, 0].plot(adaptive.frames_per_video, adaptive.selected_original_frames, marker="o", color="#278245", label="Adaptive: minimum N, maximum 20")
    axes[1, 0].set(title="Unique selected original frames (lab-matched)", xlabel="Fixed N / adaptive minimum N", ylabel="Frames")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=.2)
    axes[1, 1].hist(short.valid_face_frames, bins=np.arange(0, max(21, int(short.valid_face_frames.max()) + 2)), color="#64737C")
    axes[1, 1].set(title=f"Completed crops failing current20 | n={len(short)}", xlabel="Valid cropped frames in the file", ylabel="Videos")
    axes[1, 1].grid(axis="y", alpha=.2)
    figure.tight_layout(); figure.savefig(OUTPUT / "figures/frame_budget_tradeoff.png", dpi=180); plt.close(figure)
    selected = summary.loc[summary.scope.eq("any_exact_lab_24h")]
    lines = ["# 224视频帧数准入审计", "", "本审计只读取已有元数据、保留帧映射和上一轮化验匹配；未改裁剪、匹配、划分或训练。", "",
             "## 当前20帧排除原因", "", "|原因|视频数|涉及患者数（组间可重叠）|", "|---|---:|---:|"]
    lines.extend(f"|{r.twenty_failure_category}|{r.videos}|{r.patients}|" for r in failures.itertuples())
    lines.extend(["", "## 已有精确数值24h化验匹配的视频", "", "固定N且保留原始帧ID最小间隔2。降N后的均匀抽样可能偶尔失去原20帧视频；lost_videos列明确记录。",
                  "", "|每视频帧数|视频|患者|新增视频|新增患者|失去原视频|抽取原始帧总数|准入视频内全部有效帧|", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in selected.loc[selected.policy.eq("current")].sort_values("frames_per_video", ascending=False).itertuples():
        lines.append(f"|{r.frames_per_video}|{r.eligible_videos}|{r.eligible_patients}|{r.added_videos_vs_current20}|{r.added_patients_vs_current20}|{r.lost_videos_vs_current20}|{r.selected_original_frames}|{r.all_retained_original_frames_in_admitted_videos}|")
    lines.extend(["", "## 两种20帧放宽", "", "|方案|视频|患者|新增视频|新增患者|抽取原始帧总数|", "|---|---:|---:|---:|---:|---:|"])
    for policy, label in (("constrained", "保留间隔2，改用约束均匀抽样"), ("gap1", "允许相邻原始帧")):
        r = selected.loc[selected.policy.eq(policy) & selected.frames_per_video.eq(20)].iloc[0]
        lines.append(f"|{label}|{r.eligible_videos}|{r.eligible_patients}|{r.added_videos_vs_current20}|{r.added_patients_vs_current20}|{r.selected_original_frames}|")
    lines.extend(["", "## 自适应帧数", "", "保留长视频20帧；短视频使用全部可选非相邻帧，最多20帧，不复制补齐。", "",
                  "|最低帧数|视频|患者|新增视频|新增患者|抽取原始帧总数|", "|---:|---:|---:|---:|---:|---:|"])
    for r in selected.loc[selected.policy.eq("adaptive_cap20")].sort_values("frames_per_video", ascending=False).itertuples():
        lines.append(f"|{r.frames_per_video}|{r.eligible_videos}|{r.eligible_patients}|{r.added_videos_vs_current20}|{r.added_patients_vs_current20}|{r.selected_original_frames}|")
    lines.extend(["", "## 当前10个训练任务的并集", "", "以下采用现有主实验的严格临床字段、单位及范围口径，不新增匹配定义。", "",
                  "|方案|视频|患者|新增视频|新增患者|抽取原始帧总数|", "|---|---:|---:|---:|---:|---:|"])
    main_tasks = summary.loc[summary.scope.eq("current10_tasks_union")]
    for policy, count, label in (("current", 20, "当前固定20帧"), ("constrained", 20, "20帧约束抽样fallback"),
                                 ("current", 10, "统一固定10帧"), ("current", 5, "统一固定5帧"),
                                 ("adaptive_cap20", 10, "自适应10-20帧"), ("adaptive_cap20", 5, "自适应5-20帧")):
        r = main_tasks.loc[main_tasks.policy.eq(policy) & main_tasks.frames_per_video.eq(count)].iloc[0]
        lines.append(f"|{label}|{r.eligible_videos}|{r.eligible_patients}|{r.added_videos_vs_current20}|{r.added_patients_vs_current20}|{r.selected_original_frames}|")
    lines.extend(["", "## 零保留视频的边界条件", "",
                  f"248个零保留视频中，{int(zero.border_only_rejected_frames.gt(0).sum())}个至少有一帧仅因扩展框越界>10%被拒；"
                  f"其中{int(zero.border_only_rejected_frames.ge(39).sum())}个有至少39个此类帧。"
                  "这只能作为复查候选，不是可以自动恢复的数量：没有保存越界比例和被拒绝框的完整轨迹，不能从计数推导改到15%/20%的结果。",
                  "建议先检查是否只是向上扩展20%的padding越界，而非真实面部缺失；不能直接允许不完整人脸或解除时间戳校验。",
                  f"5个可以重选20帧的视频已实际验证{int(recovered.decoded_frames.sum())}帧FFV1解码、224尺寸、最小原始帧间隔和置信度；未写回视频或训练索引。"])
    lines.extend(["", "## 注意", "", "- 帧间隔2指原始frame ID之差，不是裁剪后视频连续帧位置；源帧被过滤后，相邻保留帧可能本就不相邻。",
                  "- 固定N的训练帧数=N×视频数；按5-view展开再乘5，不代表5倍独立临床样本。",
                  "- 如果只把N作准入门槛而训练全部帧，使用all_retained_original_frames列，不能与固定N抽样总数混淆。",
                  "- 自适应帧数应保留等视频权重（或先视频聚合），避免帧多的视频拥有更大总loss权重；现有固定20帧代码不能直接改一个常数完成此方案。",
                  "- 降帧数不能恢复没有合格人脸帧、源文件缺失或时间戳映射错误的视频。降低0.75阈值需要重新检测/裁剪，本审计不推测可恢复数。",
                  "- 新增视频不一定新增患者或独立化验时点；必须保持患者级split，新患者才需要补充分配。",
                  "- per_video_frame_counts.csv记录短视频时间跨度、置信度和可选帧上限，供进一步质量审查。",
                  "- 当前10个任务采用其自身严格临床标签口径；any_exact_lab_24h沿用全指标统计口径，两者分别报告。"])
    (OUTPUT / "reports/FRAME_BUDGET_AUDIT.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "metadata/frame_budget_audit_manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2) + "\n")


def main():
    videos = pd.read_csv(MATCH_TABLES / "video_inventory.csv", dtype={"hospital_id": str, "video_id": str})
    matches = pd.read_csv(MATCH_TABLES / "canonical_video_matches.csv", dtype={"hospital_id": str, "video_id": str})
    metadata = json.loads((OUTPUT / "metadata/video24h_matching_manifest.json").read_text())
    if _sha256(HERE.parents[1] / "merged_lab_tests.csv") != metadata["lab_table_sha256"]:
        raise RuntimeError("The previous lab matching audit is stale")
    frames, quality = inspect_videos(videos)
    recovered = verify_recovered_twenty(frames)
    tasks = task_membership(videos)
    scopes = {"all_valid_sessions": set(videos.video_id), "any_exact_lab_24h": set(matches.video_id),
              "current10_tasks_union": set().union(*tasks.values()), **tasks}
    summary = scenario_metrics(frames, scopes)
    for target in ALL_REGRESSION_TARGETS:
        current = pd.read_csv(REFERENCE / f"task_records/{target}.csv", dtype={"video_id": str})
        actual = set(frames.loc[frames.current_n20 & frames.video_id.isin(tasks[target]), "video_id"])
        if set(current.video_id) != actual:
            raise RuntimeError(f"Main task baseline differs: {target}")
    manifests = {"read_only_protocol_audit": True, "lab_table_sha256": metadata["lab_table_sha256"],
                 "inventory_sha256": _sha256(MATCH_TABLES / "video_inventory.csv"), "code_sha256": _sha256(__file__),
                 "thresholds": THRESHOLDS, "source_minimum_gap": 2, "confidence_threshold_unchanged": .75,
                 "policies": {"current": "5%-95% quantiles, then full-span fallback", "constrained": "maximum cardinality gap>=2 subset, then uniform subselect",
                              "gap1": "same uniform policy; unique but potentially adjacent source frames", "adaptive_cap20": "require minimum gap2 capacity; use min(20,capacity)"},
                 "new_video_or_pixel_cache": False, "training_modified": False}
    publish(frames, summary, quality, manifests, recovered)
    print(frames.loc[~frames.current_n20].twenty_failure_category.value_counts().to_string(), flush=True)
    print(summary.loc[summary.scope.eq("any_exact_lab_24h") & summary.policy.eq("current")].sort_values("frames_per_video", ascending=False).to_string(index=False), flush=True)
    print(f"[complete] {OUTPUT / 'reports/FRAME_BUDGET_AUDIT.md'}", flush=True)


if __name__ == "__main__":
    main()
