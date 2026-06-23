"""Headless Streamlit interface for resumable ECG SQA annotation."""

import argparse
from datetime import datetime, timezone
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import streamlit as st

_STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STUDY_DIR not in sys.path:
    sys.path.insert(0, _STUDY_DIR)

from exp1_sqa_pretrain.sqa.raw_windows import extract_window, polarity_for_source

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ANNOTATION_ROOT = os.path.join(_PKG_DIR, "sqa_annotations")
LABEL_VALUES = ("bad", "uncertain", "good")
OVERRIDE_OPTIONS = ("inherit", "bad", "uncertain", "good")
LABEL_COLUMNS = (
    "queue_id", "overall_label", "qrs_override", "morph_override",
    "status", "artifact_tags", "notes", "annotator", "annotated_at",
    "annotation_version",
)


def _atomic_write(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _text(value):
    return "" if pd.isna(value) else str(value)


def _filtered_signal(signal, sampling_rate_hz):
    if len(signal) < 16 or sampling_rate_hz <= 2:
        return signal
    high = min(40.0, 0.45 * sampling_rate_hz)
    if high <= 0.5:
        return signal
    try:
        sos = butter(
            2, [0.5, high], btype="bandpass",
            fs=sampling_rate_hz, output="sos",
        )
        return sosfiltfilt(sos, signal)
    except ValueError:
        return signal


class AnnotationStore:
    """Small CSV-backed store with atomic updates and schema validation."""

    def __init__(self, queue_path, labels_path, annotator):
        self.queue_path = os.path.abspath(queue_path)
        self.labels_path = os.path.abspath(labels_path)
        self.annotator = annotator
        self.queue = (
            pd.read_csv(self.queue_path)
            .sort_values("display_order")
            .reset_index(drop=True)
        )
        if self.queue["queue_id"].duplicated().any():
            raise ValueError("queue_id must be unique")

        self.labels = {}
        if os.path.exists(self.labels_path):
            labels = pd.read_csv(
                self.labels_path, dtype=str, keep_default_na=False
            )
            missing = set(LABEL_COLUMNS) - set(labels.columns)
            if missing:
                raise ValueError(
                    "Existing labels use an incompatible schema. "
                    f"Missing columns: {sorted(missing)}"
                )
            unknown = set(labels["queue_id"]) - set(self.queue["queue_id"])
            if unknown:
                raise ValueError(
                    f"Labels contain unknown queue IDs: {sorted(unknown)}"
                )
            self.labels = {
                row["queue_id"]: {
                    column: _text(row[column]) for column in LABEL_COLUMNS
                }
                for _, row in labels.iterrows()
            }

    def label_for(self, queue_id):
        if queue_id not in self.labels:
            self.labels[queue_id] = {column: "" for column in LABEL_COLUMNS}
            self.labels[queue_id]["queue_id"] = queue_id
        return self.labels[queue_id]

    def _save(self):
        rows = [self.labels[key] for key in sorted(self.labels)]
        _atomic_write(pd.DataFrame(rows, columns=LABEL_COLUMNS), self.labels_path)

    def _touch(self, label):
        label["annotator"] = self.annotator
        label["annotated_at"] = datetime.now(timezone.utc).isoformat()
        label["annotation_version"] = "sqa_human_v2"

    def complete(self, queue_id, overall, qrs_override, morph_override, notes):
        if overall not in LABEL_VALUES:
            raise ValueError(f"Invalid overall label: {overall}")
        label = self.label_for(queue_id)
        label["overall_label"] = overall
        label["qrs_override"] = "" if qrs_override == "inherit" else qrs_override
        label["morph_override"] = "" if morph_override == "inherit" else morph_override
        label["notes"] = notes.strip()
        label["status"] = "complete"
        self._touch(label)
        self._save()

    def skip(self, queue_id):
        label = self.label_for(queue_id)
        label["status"] = "skipped"
        self._touch(label)
        self._save()

    def clear(self, queue_id):
        label = self.label_for(queue_id)
        for column in (
            "overall_label", "qrs_override", "morph_override", "status",
            "artifact_tags", "notes",
        ):
            label[column] = ""
        self._touch(label)
        self._save()

    def first_incomplete(self):
        for position, row in self.queue.iterrows():
            status = self.labels.get(row["queue_id"], {}).get("status")
            if status not in {"complete", "skipped"}:
                return int(position)
        return max(0, len(self.queue) - 1)

    def next_incomplete(self, position):
        positions = list(range(position + 1, len(self.queue)))
        positions += list(range(0, position + 1))
        for candidate in positions:
            queue_id = self.queue.iloc[candidate]["queue_id"]
            status = self.labels.get(queue_id, {}).get("status")
            if status not in {"complete", "skipped"}:
                return candidate
        return position

    def counts(self):
        completed = sum(
            label.get("status") == "complete" for label in self.labels.values()
        )
        skipped = sum(
            label.get("status") == "skipped" for label in self.labels.values()
        )
        return completed, skipped


@st.cache_data(show_spinner=False, max_entries=12)
def _load_window(
    file_path,
    start_time,
    window_sec,
    target_length,
    data_source,
    polarity,
    corruption_type,
    corruption_severity,
    corruption_seed,
):
    return extract_window(
        file_path,
        start_time,
        window_sec=window_sec,
        target_length=target_length,
        data_source=data_source,
        polarity=polarity,
        corruption_type=corruption_type,
        corruption_severity=corruption_severity,
        corruption_seed=corruption_seed,
    )


def _row_value(row, column, default):
    value = row.get(column, default)
    return default if pd.isna(value) or value == "" else value


def _reset_form_state(queue_id):
    for suffix in ("qrs", "morph", "notes"):
        st.session_state.pop(f"{queue_id}_{suffix}", None)


def _chart_frames(window, window_sec, target_length):
    timestamps = window["timestamps"]
    relative_time = timestamps - timestamps[0]
    raw = pd.DataFrame(
        {"Raw ECG": window["raw_ecg"]},
        index=pd.Index(relative_time, name="Time (s)"),
    )

    normalized = window["model_input"]
    target_time = np.arange(target_length) * (window_sec / target_length)
    filtered = _filtered_signal(normalized, target_length / window_sec)
    processed = pd.DataFrame(
        {"Normalized": normalized, "0.5–40 Hz": filtered},
        index=pd.Index(target_time, name="Time (s)"),
    )
    return raw, processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless Streamlit ECG SQA annotation app"
    )
    parser.add_argument("--round-id", default="round01")
    parser.add_argument("--queue", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--annotator", default=os.environ.get("USER", "annotator"))
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--show-predictions", action="store_true")
    args, _ = parser.parse_known_args()
    round_dir = os.path.join(DEFAULT_ANNOTATION_ROOT, args.round_id)
    args.queue = args.queue or os.path.join(round_dir, "queue.csv")
    args.labels = args.labels or os.path.join(round_dir, "labels.csv")
    return args


def run_app(args):
    st.set_page_config(
        page_title="ECG SQA Annotation", page_icon="🫀", layout="wide"
    )
    st.title("ECG Signal Quality Annotation")

    try:
        store = AnnotationStore(args.queue, args.labels, args.annotator)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    if "annotation_position" not in st.session_state:
        st.session_state.annotation_position = store.first_incomplete()
    position = int(np.clip(
        st.session_state.annotation_position, 0, len(store.queue) - 1
    ))
    st.session_state.annotation_position = position
    row = store.queue.iloc[position]
    queue_id = row["queue_id"]
    label = store.label_for(queue_id)
    completed, skipped = store.counts()

    with st.sidebar:
        st.subheader("Session")
        st.write(f"Round: `{args.round_id}`")
        st.write(f"Annotator: `{args.annotator}`")
        st.write(f"Completed: **{completed}/{len(store.queue)}**")
        st.write(f"Skipped: **{skipped}**")
        st.progress(completed / max(1, len(store.queue)))
        if st.button("Go to first pending", width="stretch"):
            st.session_state.annotation_position = store.first_incomplete()
            st.rerun()
        st.caption(
            "Model scores, candidate source, and repeat status are hidden by default."
        )

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader(f"Sample {position + 1} of {len(store.queue)}")
    with header_right:
        status = label.get("status") or "pending"
        st.markdown(f"Status: **{status}**")

    try:
        data_source = _row_value(row, "data_source", "raw")
        default_polarity = polarity_for_source(data_source)
        window = _load_window(
            row["file_path"], float(row["start_time"]),
            float(args.window_sec), int(args.target_length),
            data_source,
            float(_row_value(row, "polarity", default_polarity)),
            str(_row_value(row, "corruption_type", "none")),
            int(float(_row_value(row, "corruption_severity", 0))),
            int(float(_row_value(row, "corruption_seed", 0))),
        )
    except (OSError, ValueError) as error:
        st.error(f"Could not load this ECG window: {error}")
        st.stop()

    raw_frame, processed_frame = _chart_frames(
        window, args.window_sec, args.target_length
    )
    st.caption("Raw ECG")
    st.line_chart(raw_frame, height=260, width="stretch")
    st.caption("Normalized ECG and filtered view")
    st.line_chart(processed_frame, height=260, width="stretch")

    if args.show_predictions and pd.notna(row.get("tcn_window_p_qrs")):
        st.info(
            "TCN QRS/Morph: "
            f"{row['tcn_window_p_qrs']:.3f} / {row['tcn_window_p_morph']:.3f} · "
            "ResNet QRS/Morph: "
            f"{row['resnet_window_p_qrs']:.3f} / {row['resnet_window_p_morph']:.3f}"
        )

    qrs_default = label.get("qrs_override") or "inherit"
    morph_default = label.get("morph_override") or "inherit"
    with st.form(f"annotation_{queue_id}"):
        override_left, override_right = st.columns(2)
        with override_left:
            qrs_override = st.selectbox(
                "QRS override (optional)",
                OVERRIDE_OPTIONS,
                index=OVERRIDE_OPTIONS.index(qrs_default),
                key=f"{queue_id}_qrs",
            )
        with override_right:
            morph_override = st.selectbox(
                "Morphology override (optional)",
                OVERRIDE_OPTIONS,
                index=OVERRIDE_OPTIONS.index(morph_default),
                key=f"{queue_id}_morph",
            )
        notes = st.text_input(
            "Notes (optional)", value=label.get("notes", ""),
            key=f"{queue_id}_notes",
        )
        bad_col, uncertain_col, good_col = st.columns(3)
        save_bad = bad_col.form_submit_button(
            "Save BAD", type="secondary", width="stretch"
        )
        save_uncertain = uncertain_col.form_submit_button(
            "Save UNCERTAIN", type="secondary", width="stretch"
        )
        save_good = good_col.form_submit_button(
            "Save GOOD", type="primary", width="stretch"
        )

    selected = (
        "bad" if save_bad else
        "uncertain" if save_uncertain else
        "good" if save_good else None
    )
    if selected is not None:
        store.complete(
            queue_id, selected, qrs_override, morph_override, notes
        )
        st.session_state.annotation_position = store.next_incomplete(position)
        st.rerun()

    nav_previous, nav_next, nav_skip, nav_clear = st.columns(4)
    if nav_previous.button("Previous", width="stretch"):
        st.session_state.annotation_position = max(0, position - 1)
        st.rerun()
    if nav_next.button("Next", width="stretch"):
        st.session_state.annotation_position = min(len(store.queue) - 1, position + 1)
        st.rerun()
    if nav_skip.button("Skip", width="stretch"):
        store.skip(queue_id)
        st.session_state.annotation_position = store.next_incomplete(position)
        st.rerun()
    if nav_clear.button("Clear current", width="stretch"):
        store.clear(queue_id)
        _reset_form_state(queue_id)
        st.rerun()

    current_overall = label.get("overall_label") or "—"
    current_qrs = label.get("qrs_override") or "inherit"
    current_morph = label.get("morph_override") or "inherit"
    st.caption(
        f"Saved label: overall={current_overall}, "
        f"QRS={current_qrs}, Morph={current_morph}"
    )


if __name__ == "__main__":
    run_app(parse_args())
