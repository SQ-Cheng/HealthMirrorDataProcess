"""Focused checks for source-row coverage and patient mixing."""

import unittest

import pandas as pd
import torch

from .data import PatientDiversePairSampler


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


if __name__ == "__main__":
    unittest.main()
