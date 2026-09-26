"""Focused tests for the patient-diverse frame-order ablation."""

import unittest

import numpy as np
import pandas as pd
import torch

from .data import InterleavedPatientViewBatchSampler, PatientDiverseFrameSampler


class PatientDiverseFrameSamplerTest(unittest.TestCase):
    def test_all_frames_once_with_diverse_patients(self):
        class DatasetStub:
            expand_all_views = True
            video_records = pd.DataFrame({
                "hospital_id": [f"patient_{index // 2}" for index in range(48)]
            })
            frame_video_rows = np.repeat(np.arange(48), 20)
            frame_count = len(frame_video_rows)

            def __len__(self):
                return len(self.frame_video_rows)

        dataset = DatasetStub()
        sampler = PatientDiverseFrameSampler(dataset, source_batch_size=48)
        torch.manual_seed(123)
        order = list(sampler)
        self.assertEqual(len(order), len(dataset))
        self.assertEqual(sorted(order), list(range(len(dataset))))
        batches = [order[index:index + 48] for index in range(0, len(order), 48)]
        self.assertTrue(all(len(batch) == 48 for batch in batches))
        unique_patients = [len({
            dataset.video_records.iloc[dataset.frame_video_rows[index]].hospital_id
            for index in batch
        }) for batch in batches]
        self.assertGreaterEqual(np.median(unique_patients), 11)
        torch.manual_seed(123)
        self.assertEqual(order, list(sampler))

    def test_views_are_spread_across_distinct_frame_batches(self):
        class DatasetStub:
            expand_all_views = False
            views = ("original", "hflip", "center_crop", "brightness", "contrast")
            video_records = pd.DataFrame({
                "hospital_id": [f"patient_{index // 2}" for index in range(48)]
            })
            frame_video_rows = np.repeat(np.arange(48), 20)
            frame_count = len(frame_video_rows)

            def __len__(self):
                return self.frame_count * len(self.views)

        dataset = DatasetStub()
        sampler = InterleavedPatientViewBatchSampler(dataset, batch_size=240)
        torch.manual_seed(123)
        batches = list(sampler)
        indices = [index for batch in batches for index in batch]
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(indices), list(range(len(dataset))))
        self.assertTrue(all(len(batch) == 240 for batch in batches))
        self.assertTrue(all(len({index // 5 for index in batch}) == 240
                            for batch in batches))
        self.assertEqual(np.bincount(np.asarray(indices) % 5).tolist(),
                         [dataset.frame_count] * 5)


if __name__ == "__main__":
    unittest.main()
