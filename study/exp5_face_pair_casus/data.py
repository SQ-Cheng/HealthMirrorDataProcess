"""Paired-frame dataset using normalized partial-CASUS targets."""

from study.exp5_face_pair_recovery.data import PairedFrameDataset, SingleFrameDataset

from .config import CASUS_MAX_SCORE


class PairedCasusDataset(PairedFrameDataset):
    def __init__(self, frame_index, records, views=("original",), expand_views=False):
        normalized = records.copy()
        normalized["recovery_score"] = normalized["casus_score"] / CASUS_MAX_SCORE
        super().__init__(frame_index, normalized, views, expand_views)


class PostOnlyCasusDataset(SingleFrameDataset):
    def __init__(self, frame_index, records, views=("original",), expand_views=False):
        normalized = records.copy()
        normalized["recovery_score"] = normalized["casus_score"] / CASUS_MAX_SCORE
        # SingleFrameDataset only consumes post_indices, but its shared index
        # initialization expects a paired column. Self-pairing avoids any
        # preoperative-video requirement or decode.
        normalized["pre_video_id"] = normalized["video_id"]
        super().__init__(
            frame_index, normalized, "post_only", views, expand_views,
        )
