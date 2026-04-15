Files:
- STATE_LOCAL.md
- AGENT.md
- PROTOCOL.md
- task_rlvr_tiny_transformer.md
- arithmetic-transformer/model.py
- arithmetic-transformer/rlvr_tiny/
- arithmetic-transformer/TASK_RESULTS.md

Curent task [4]:
Build the modular RLVR tiny-transformer framework from the task file, run small validation experiments phase by phase, and save a concise task report.

Goal task:
- [done] Rewrite `AGENT.md` into a clear working instruction file.
- [done] Create `STATE_LOCAL.md` with the initial local task state.
- [done] Add the cloned upstream project files into this workspace.
- [done] Identify the LSTM experiment entry point and 32K-scale defaults.
- [done] Prepare and run CPU LSTM experiments for 1-digit and 3-digit addition.
- [done] Document results in a short markdown file.
- [done] Add checkpoint saving for the curriculum run.
- [done] Add workspace rules for per-digit checkpoints and protocol logging.
- [done] Add workspace rules for a slim Colab-ready execution bundle.
- [done] Replace duplicated runtime copies with one original runnable folder: `arithmetic-transformer/`.
- [done] Put the notebook in `arithmetic-transformer/` with both local and Colab paths.
- [done] Do one local test run and verify local checkpoint saving.
- [done] Add checkpoint validation script and log validation result.
- [done] Initialize a root git repository for this workspace.
- [done] Add `.gitignore` rule for model checkpoints (`*.pt` only).
- [done] Implement symbolic formatting, dataset generation, verifier, curriculum config, and evaluation modules for RLVR arithmetic traces.
- [done] Implement short SFT, RLVR, and best-of-N training loops for tiny transformer phases.
- [done] Run small validation experiments for early phases and baseline comparisons.
- [done] Write `TASK_RESULTS.md` with literature summary, metrics, examples, and recommendations.
