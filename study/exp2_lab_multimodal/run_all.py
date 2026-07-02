"""End-to-end pipeline for Exp2: Multi-Modal Deep Learning for Lab Test Prediction.

Steps:
    1. Build dataset (extract ECG + Face features, match lab labels)
    2. Train & evaluate multi-modal deep learning model
    3. Generate report

Usage:
    python -m study.exp2_lab_multimodal.run_all
    python -m study.exp2_lab_multimodal.run_all --skip-build  # skip dataset building
"""

import argparse
import os
import sys
import time

from .config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Exp2: Multi-modal deep learning for lab test prediction"
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip dataset building (use existing features.npz)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (for quick testing)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max training epochs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    # ── Step 1: Build dataset ────────────────────────────────────────
    if not args.skip_build:
        print("\n" + "=" * 60)
        print("STEP 1: Building Dataset")
        print("=" * 60)
        from .build_dataset import build_features
        manifest, ecg, face = build_features(
            output_dir=args.output_dir, max_samples=args.max_samples
        )
    else:
        import numpy as np
        import pandas as pd
        features_path = os.path.join(args.output_dir, "features.npz")
        manifest_path = os.path.join(args.output_dir, "manifest.csv")
        if not os.path.exists(features_path) or not os.path.exists(manifest_path):
            print("ERROR: --skip-build specified but features.npz/manifest.csv not found.")
            print("Run without --skip-build first.")
            sys.exit(1)
        print("Skipping dataset build (using existing features).")
        manifest = pd.read_csv(manifest_path, dtype=str)
        data = np.load(features_path, allow_pickle=True)
        ecg = data["ecg"]
        face = data["face"]

    print(f"\nDataset: {len(manifest)} samples, "
          f"{manifest['hospital_id'].nunique()} patients")

    # ── Step 2: Train & Evaluate ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training & Evaluation")
    print("=" * 60)
    from .train_eval import train_and_evaluate
    metrics_df, predictions_df = train_and_evaluate(
        manifest, ecg, face, output_dir=args.output_dir
    )

    # ── Step 3: Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Summary")
    print("=" * 60)
    elapsed = time.time() - t_start
    print(f"Total elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}/")
    print(f"  - metrics.csv")
    print(f"  - predictions.csv")
    print(f"  - split.json")
    print(f"  - checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
