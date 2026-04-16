# Experiment 1

This folder contains the first compact RLVR curriculum experiment for the tiny arithmetic transformer.

Contents:
- `config.json`: phase and model settings
- `train_experiment_1.py`: training entry point
- `logs/`: generated summaries and metrics
- `checkpoints/`: generated model checkpoints (ignored by git if `.pt`)

Goal:
- train a mini model step by step from 1-digit to 5-digit addition
- do a short smoke pass first with 1 epoch per phase
- keep logs and configs in one place
