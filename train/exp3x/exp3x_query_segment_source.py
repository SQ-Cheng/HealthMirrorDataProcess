"""Query source metadata of an Exp3/Exp3X data segment from sample number."""

import argparse
import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
EXP3_DIR = os.path.join(ROOT_DIR, "train", "exp3")

sys.path.insert(0, EXP3_DIR)

try:
    from train.exp3.exp3_dataloader import MaskedReconDataset
except ModuleNotFoundError:
    from exp3_dataloader import MaskedReconDataset  # type: ignore[import-not-found]


def parse_args():
    parser = argparse.ArgumentParser(description="Query source metadata by sample number")
    parser.add_argument("--sample-number", type=int, required=True, help="Sample number to query")
    parser.add_argument(
        "--data-source",
        choices=["sqi", "cleaned"],
        default="sqi",
        help="Use mirror*_auto_cleaned_sqi (sqi) or mirror*_auto_cleaned (cleaned)",
    )
    parser.add_argument(
        "--sample-space",
        choices=["dataset", "train", "val"],
        default="dataset",
        help="How to interpret sample number",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--target-length", type=int, default=256)
    parser.add_argument("--max-windows-per-patient", type=int, default=None)
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def build_split_indices(dataset, val_ratio, seed):
    rng = np.random.default_rng(seed)
    unique_pids = list(dict.fromkeys(dataset.hospital_pids))
    rng.shuffle(unique_pids)
    if not unique_pids:
        raise RuntimeError("No samples available.")

    n_val_pids = max(1, int(len(unique_pids) * val_ratio))
    val_pids = set(unique_pids[:n_val_pids])
    train_pids = set(unique_pids[n_val_pids:])

    train_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in train_pids]
    val_indices = [i for i, p in enumerate(dataset.hospital_pids) if p in val_pids]
    return train_indices, val_indices


def resolve_dataset_index(args, dataset):
    if args.sample_space == "dataset":
        if args.sample_number < 0 or args.sample_number >= len(dataset):
            raise IndexError(f"sample-number out of range for dataset: [0, {len(dataset) - 1}]")
        return args.sample_number, None

    train_indices, val_indices = build_split_indices(dataset, args.val_ratio, args.seed)
    if args.sample_space == "train":
        if args.sample_number < 0 or args.sample_number >= len(train_indices):
            raise IndexError(f"sample-number out of range for train split: [0, {len(train_indices) - 1}]")
        return train_indices[args.sample_number], {"train_size": len(train_indices), "val_size": len(val_indices)}

    if args.sample_number < 0 or args.sample_number >= len(val_indices):
        raise IndexError(f"sample-number out of range for val split: [0, {len(val_indices) - 1}]")
    return val_indices[args.sample_number], {"train_size": len(train_indices), "val_size": len(val_indices)}


def main():
    args = parse_args()

    dataset = MaskedReconDataset(
        ROOT_DIR,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        target_length=args.target_length,
        data_source=args.data_source,
        max_windows_per_patient=args.max_windows_per_patient,
        max_patients=args.max_patients,
    )

    dataset_idx, split_info = resolve_dataset_index(args, dataset)
    record = dict(dataset.get_source_record(dataset_idx))

    record["query"] = {
        "sample_space": args.sample_space,
        "sample_number": args.sample_number,
        "resolved_dataset_index": int(dataset_idx),
        "dataset_size": len(dataset),
    }
    record["quality"] = {
        "clean_score": float(dataset.clean_score[dataset_idx]),
        "ecg_autocorr_sqi": float(dataset.ecg_quality[dataset_idx]),
        "rppg_snr_db": float(dataset.rppg_snr_db[dataset_idx]),
    }
    if split_info is not None:
        record["split"] = split_info

    if args.pretty:
        print(json.dumps(record, indent=2))
    else:
        print(json.dumps(record))


if __name__ == "__main__":
    main()
