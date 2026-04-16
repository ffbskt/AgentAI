# AGENT.md

## Main Rule
- Keep only one original runnable code folder: `arithmetic-transformer/`.
- Put the notebook in the same folder as the runnable modules.
- Push the whole workspace to git after meaningful completed steps.

## Workspace Structure
- Root files are for coordination only:
  - `AGENT.md`
  - `STATE_LOCAL.md`
  - `PROTOCOL.md`
- All runnable training code must stay in:
  - `arithmetic-transformer/`
- Do not keep duplicate runnable copies of `train.py`, `dataset.py`, `model.py`, or `methods.py` in other folders.

## Experiment Rules
- Use the existing environment when possible.
- Read only the files needed for the current task.
- Keep experiments simple and reproducible.
- Save a checkpoint each time a new digit length is learned.
- Save protocol records for each learned stage.

## Protocol Rules
- For every successful learned stage, log:
  - model name
  - config
  - digit length
  - validation accuracy
  - total learning time from run start
  - checkpoint path

## Notebook Rules
- The notebook must live in `arithmetic-transformer/`.
- The notebook must contain both:
  - local computer paths
  - Google Colab paths
- The notebook should make it easy to switch between local and Colab execution.
- For Colab, assume the user updates code locally and pushes to git.
- The notebook must therefore clone the repo if missing and pull the latest changes if it already exists.
- After any notebook or module change that Colab depends on, commit and push before using Colab clone/pull.
- Otherwise Colab may load an old repo version even if local files are already updated.
- If notebook work for Colab is completed successfully, commit and push automatically without asking again.
- The notebook must save:
  - checkpoints
  - protocol log
  - train log
- If the repo path exists in Colab but is not a git repo, remove that broken folder and clone again.

## Path Rules For Notebook
- Local path example:
  - `D:/Codex projects/Transformer_Graph3/arithmetic-transformer`
- Colab path example:
  - repo root: `/content/drive/MyDrive/AgentAI`
  - code folder after clone: `/content/drive/MyDrive/AgentAI/arithmetic-transformer`
- Output folders should be created inside the selected repo path:
  - `checkpoints/`
  - `runs/`

## Execution Rule
- After changing notebook or training-save logic, do one small local test run.
- The test run must verify that a checkpoint is actually written locally.

## Commit Rule
- Commit completed task steps with:
  - `[project_id]_[goal_task_id]_[curent_task_id]`
