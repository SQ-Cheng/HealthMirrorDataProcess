"""Run the non-adjacent 20-frame RGB augmentation experiment."""

import argparse
import os
import sys
import time

from .config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Exp2 Aug20 non-adjacent RGB per-task experiment")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    if not args.skip_build:
        print("\n" + "=" * 60)
        print("STEP 1: Build Aug20 Non-Adjacent RGB Dataset")
        print("=" * 60)
        from .build_dataset import build_features
        manifest, face = build_features(output_dir=args.output_dir, max_samples=args.max_samples)
    else:
        import numpy as np
        import pandas as pd
        features_path = os.path.join(args.output_dir, "features.npz")
        manifest_path = os.path.join(args.output_dir, "manifest.csv")
        if not os.path.exists(features_path) or not os.path.exists(manifest_path):
            print("ERROR: --skip-build specified but features.npz/manifest.csv not found.")
            sys.exit(1)
        print("Skipping dataset build.")
        manifest = pd.read_csv(manifest_path, dtype=str)
        face = np.load(features_path, allow_pickle=True)["face"]

    print(f"\nDataset: {len(manifest)} samples, {manifest['hospital_id'].nunique()} patients")
    print("\n" + "=" * 60)
    print("STEP 2: Train/Evaluate Per-Task RGB Models on Augmented Frame Samples")
    print("=" * 60)
    from .train_eval import train_and_evaluate
    train_and_evaluate(manifest, face, output_dir=args.output_dir)
    print(f"\nTotal elapsed time: {(time.time() - start) / 60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
