"""Small, label-free checks for continuous clip selection and video sampling."""

from io import BytesIO
import unittest

from PIL import Image
import numpy as np
import pandas as pd
import torch

from .clips import (OneClipPerVideoSampler, PatientDiverseClipBatchSampler,
                    _jpeg_ranges, _select_clips)
from .config import CLIP_FRAMES


class ClipPolicyTest(unittest.TestCase):
    def test_selected_windows_are_decodable_contiguous_and_disjoint(self):
        stream = bytearray()
        for number in range(60):
            handle = BytesIO()
            Image.new("RGB", (128, 128), (number, 30, 70)).save(handle, format="JPEG")
            stream.extend(handle.getvalue())
        ranges = _jpeg_ranges(bytes(stream))
        starts = _select_clips(bytes(stream), ranges)
        self.assertEqual(len(ranges), 60)
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(b - a >= CLIP_FRAMES for a, b in zip(starts, starts[1:])))

    def test_sampler_uses_one_clip_per_video_without_duplicates(self):
        class Dataset:
            records = pd.DataFrame({"video_id": ["a", "b", "c"]})
            video_clips = [[0, 1, 2], [3], [4, 5]]

        torch.manual_seed(5)
        selected = list(OneClipPerVideoSampler(Dataset()))
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertEqual(sum(value in range(3) for value in selected), 1)

    def test_middle_48_frame_window(self):
        stream = bytearray()
        for number in range(100):
            handle = BytesIO()
            Image.new("RGB", (128, 128), (number, 30, 70)).save(handle, format="JPEG")
            stream.extend(handle.getvalue())
        starts = _select_clips(bytes(stream), _jpeg_ranges(bytes(stream)), 48, (0.5,))
        self.assertEqual(starts, [26])

    def test_patient_diverse_batches_preserve_each_video(self):
        class Dataset:
            records = pd.DataFrame({"hospital_id": ["a", "a", "b", "c", "d"]})
            video_clips = [[0, 1], [2, 3], [4], [5], [6]]

        torch.manual_seed(7)
        batches = list(PatientDiverseClipBatchSampler(Dataset(), 4))
        chosen = [clip for batch in batches for clip in batch]
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len(set(chosen)), 5)
        first_batch_patients = [Dataset.records.hospital_id.iloc[
            next(row for row, clips in enumerate(Dataset.video_clips) if clip in clips)
        ] for clip in batches[0]]
        self.assertEqual(len(first_batch_patients), len(set(first_batch_patients)))


if __name__ == "__main__":
    unittest.main()
