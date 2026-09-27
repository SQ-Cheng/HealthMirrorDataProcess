"""Focused checks for source-row coverage and patient mixing."""

import unittest

import pandas as pd
import torch

from .config import PATIENT_DIVERSE_30_40, SCHEDULE_ONLY_30_40
from .data import ChunkShuffleSampler, PatientDiversePairSampler


class _Dataset:
    def __init__(self, patient_ids):
        self.records = pd.DataFrame({"hospital_id": patient_ids})

    def __len__(self):
        return 20 * len(self.records)


class PatientDiversePairSamplerTest(unittest.TestCase):
    def test_every_frame_once_and_diverse_batches(self):
        dataset = _Dataset([f"p{i:02d}" for i in range(18)] * 2)
        torch.manual_seed(31)
        order = list(PatientDiversePairSampler(dataset, source_batch_size=24))
        self.assertEqual(sorted(order), list(range(len(dataset))))
        full_batches = [order[i:i + 24] for i in range(0, len(order) - 23, 24)]
        unique_patients = [len({dataset.records.iloc[row // 20].hospital_id
                                for row in batch}) for batch in full_batches]
        self.assertGreaterEqual(sum(count == 12 for count in unique_patients),
                                int(len(full_batches) * 0.8))
        torch.manual_seed(31)
        self.assertEqual(order, list(PatientDiversePairSampler(dataset, 24)))

    def test_rejects_invalid_group_size(self):
        with self.assertRaises(ValueError):
            PatientDiversePairSampler(_Dataset(["p0"]), 24, frames_per_group=3)

    def test_schedule_only_keeps_chunked_pair_batches(self):
        self.assertEqual(
            {key: value for key, value in SCHEDULE_ONLY_30_40.items()
             if key != "train_batch_policy"},
            {key: value for key, value in PATIENT_DIVERSE_30_40.items()
             if key != "train_batch_policy"},
        )
        self.assertEqual(SCHEDULE_ONLY_30_40["train_batch_policy"], "chunk")
        dataset = _Dataset([f"p{i:02d}" for i in range(6)])
        torch.manual_seed(7)
        order = list(ChunkShuffleSampler(dataset))
        self.assertEqual(sorted(order), list(range(len(dataset))))
        for start in range(0, len(order), 24):
            batch = order[start:start + 24]
            if len(batch) == 24:
                self.assertEqual(len({row // 20 for row in batch}), 2)


if __name__ == "__main__":
    unittest.main()
