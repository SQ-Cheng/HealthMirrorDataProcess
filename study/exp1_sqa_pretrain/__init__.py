"""Exp1-SQAPreTrain: Self-supervised masked signal reconstruction pre-training.

Unified framework for training masked reconstruction models on physiological signals
(ECG, rPPG, or joint ECG+rPPG).

Directory structure:
    exp1_sqa_pretrain/
    ├── models/         # Model architectures (baseline, tcn, mamba, transformer, gan, joint)
    ├── checkpoints/    # Saved model checkpoints
    ├── plots/          # Training history and visualization plots
    ├── dataloader.py   # Data loading and preprocessing
    ├── train.py        # Training script
    ├── eval.py         # Evaluation script
    └── visualize.py    # Quick visualization script

Usage:
    # Train a baseline model on ECG
    python -m exp1_sqa_pretrain.train --model baseline --variant light --signal-type ecg

    # Train a TCN with curriculum learning
    python -m exp1_sqa_pretrain.train --model tcn --variant tcn256 --signal-type ecg --curriculum

    # Evaluate a trained model
    python -m exp1_sqa_pretrain.eval --model tcn --variant tcn256 --signal-type ecg

    # Visualize reconstructions
    python -m exp1_sqa_pretrain.visualize --model baseline --variant full --signal-type ecg
"""
