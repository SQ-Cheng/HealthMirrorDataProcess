#!/usr/bin/env python
"""Run Direction A: ECG Feature Engineering for Lab Abnormality Prediction."""

import os
import sys

import numpy as np

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from study.exp3_independent_research.config import OUTPUT_DIR, SEED
from study.exp3_independent_research.direction_a_feature_eng.extract_features import (
    build_feature_dataset,
)
from study.exp3_independent_research.direction_a_feature_eng.train_eval import (
    train_and_evaluate,
)


def main():
    print("=" * 60)
    print("Exp3 — Direction A: ECG Feature Engineering")
    print("=" * 60)

    # Step 1: Build feature dataset
    print("\nStep 1: Extracting ECG features...")
    features_df = build_feature_dataset()
    print(f"  Extracted {len(features_df)} samples with {features_df.shape[1]} columns")

    # Step 2: Train and evaluate
    print("\nStep 2: Training classical ML models...")
    results_df, importance_df = train_and_evaluate(features_df)

    print("\nDone! Results saved to outputs/direction_a/")


if __name__ == "__main__":
    main()
