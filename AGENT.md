# AGENT.md

## Purpose
This repository is a workspace for running and documenting experiments based on `thomasahle/arithmetic-transformer`.

The main focus is:
- use the cloned upstream project inside this repo;
- run experiments only for Transformer, LSTM, and Sinc variants;
- target 32K and 120K settings;
- keep execution CPU-friendly;
- produce simple markdown notes and toy experiments with clear, reproducible results.

## Environment
- Preferred virtual environment: `D:\Codex projects\.venv`
- Do not create a new environment unless explicitly requested.
- Reuse the existing environment and update it only when necessary.

## Working Rules
- Read only the files needed for the current task.
- Keep token and time usage low.
- Do not scan the whole cloned repository by default.
- When useful, leave short markdown notes that help future work.
- Prefer small, verifiable steps over large untracked changes.
- Keep experiment setup simple and easy to rerun on CPU.
- For curriculum experiments, save a checkpoint each time the model successfully learns a digit length.
- Do not treat a run as complete if checkpoints for the learned stages were not saved.
- Log each successful stage with model name, config, digit length, accuracy, and total learning time.
- Keep a slim runnable bundle in the repo for external execution environments such as Google Colab.
- That bundle must contain only the required execution files, not the whole research workspace.
- Prefer rebuilding and updating the slim bundle when training code changes.

## Experiment Goals
1. Prepare a minimal local copy/setup of the arithmetic-transformer project in this workspace.
2. Identify the relevant code and configs for Transformer, LSTM, and Sinc experiments.
3. Create simple toy experiments for 32K and 120K settings.
4. Record results in markdown with enough detail to reproduce them.
5. Keep conclusions clear, short, and directly tied to the observed outputs.

## State Management
- Use `STATE_LOCAL.md` to track the current working step.
- If `STATE_LOCAL.md` is missing or finished, create/update it before starting code work.
- Keep tasks split into small current steps with file targets.
- Update the current step after each successful change. Edit Files list to do not read unneccessary files.

## Commit Rule
- If this workspace is inside a git repository, commit each completed current task with:
  `[project_id]_[goal_task_id]_[current_task_id]`
- If git is not initialized yet, continue the task and note that commits are blocked until the repo is initialized.

## Output Expectations
- Create short markdown logs for experiments.
- Prefer concrete outputs over long theory.
- Each experiment note should include:
  - what was run;
  - key parameters;
  - CPU-related constraints if relevant;
  - observed result;
  - short conclusion.

## Checkpoint And Protocol Rule
- For staged learning runs, save checkpoints for each learned stage: 1-digit, 2-digit, 3-digit, and so on.
- Save checkpoints in the experiment workspace, not only the final model.
- Maintain a protocol log for every learned stage.
- Each protocol record must contain:
  - model name;
  - full config or command settings;
  - learned digit length;
  - validation accuracy at success;
  - total learning time from run start;
  - checkpoint path.

## Colab Bundle Rule
- Maintain a folder such as `colab_bundle/` with only the files needed to run training in Colab.
- Minimum expected runnable files:
  - `train.py`
  - `dataset.py`
  - `model.py`
  - `methods.py`
  - minimal requirements file
  - small Colab notebook
  - short README for Colab usage
- The notebook should:
  - mount Google Drive;
  - install the minimal dependencies;
  - run training from the slim bundle;
  - save checkpoints and protocol logs to Drive.
- If the main training code changes, update the Colab bundle in the same task when possible.

## Default Task For This Workspace
Set up the project so future turns can quickly run and document toy experiments for Transformer, LSTM, and Sinc at 32K and 120K on CPU.
