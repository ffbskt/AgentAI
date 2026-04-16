Files:
- STATE_LOCAL.md
- AGENT.md
- PROTOCOL.md
- task_rlvr_tiny_transformer.md
- arithmetic-transformer/rlvr_tiny/verify.py
- arithmetic-transformer/rlvr_tiny/tests/

Curent task [4]:
Create synthetic verifier corner cases, test each verifier function on them, and run the tests locally.

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
- [done] Add two synthetic corner-case model outputs: one correct and one incorrect.
- [done] Add focused verifier tests that use those synthetic outputs with each verifier function.
- [done] Run the tests and confirm the verifier behaves correctly.
