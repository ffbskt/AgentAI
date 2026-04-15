# PROTOCOL.md

## Purpose
This file defines the protocol structure for experiment logging in this workspace.

## Required Checkpoint Policy
- For curriculum or staged experiments, save a checkpoint every time a new digit length is learned successfully.
- Minimum expectation for addition runs:
  - 1-digit checkpoint
  - 2-digit checkpoint
  - 3-digit checkpoint
- Continue the same rule for higher digit lengths.

## Required Protocol Record Structure
Each successful learned stage must be logged as one record with:
- `event`: usually `digit_learned`
- `model`: model type such as `lstm`, `transformer`, `transformer-lstm`
- `config`: the run config used for the experiment
- `digit`: learned digit length
- `accuracy`: validation accuracy at the success point
- `epochs_per_digit`: current curriculum progress
- `total_learning_time_seconds`: elapsed time from the beginning of the run
- `total_learning_time_minutes`: same value in minutes
- `checkpoint_path`: saved model path for this stage

## Output Format
- Preferred machine-readable format: JSON Lines (`.jsonl`)
- One line per successful learned stage
- Human-readable summary may also be written in markdown, but the protocol log is required

## Minimum Human Summary
For each completed experiment, also leave a short human summary with:
- model
- key config
- which digits were learned
- where checkpoints were saved
- total training time to each learned digit
