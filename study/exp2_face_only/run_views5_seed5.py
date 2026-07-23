"""Build native-resolution 24-hour Aug20 data and train hemoglobin tasks."""

import argparse
import os
import time

from .config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()

    if not args.skip_build:
        print(
            "STEP 1: Build native-resolution Aug20 data with a 24-hour lab-video window",
            flush=True,
        )
        from .build_dataset import build_features
        build_features(output_dir=args.output_dir)

    print("STEP 2: Dynamic five-view, two-seed hemoglobin-low training", flush=True)
    from . import train_views5_seed5 as trainer
    argv = [
        "--input-dir", args.output_dir,
        "--workers", str(args.workers),
    ]
    if args.seeds:
        argv += ["--seeds", args.seeds]
    if args.tasks:
        argv += ["--tasks", args.tasks]
    if args.max_epochs is not None:
        argv += ["--max-epochs", str(args.max_epochs)]
    import sys
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        trainer.main()
    finally:
        sys.argv = previous
    print(f"Total elapsed minutes: {(time.time() - start) / 60.0:.1f}", flush=True)


if __name__ == "__main__":
    main()
