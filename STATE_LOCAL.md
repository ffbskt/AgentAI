Files:
- STATE_LOCAL.md
- AGENT.md
- PROTOCOL.md
- arithmetic-transformer/experiment_1/
- arithmetic-transformer/rlvr_tiny/

Curent task [7]:
Harden `experiment_1/run_experiment_1.ipynb` imports against stale notebook kernels by forcing a fresh `train_experiment_1` load and printing resolved module details.

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
- [todo] Create `experiment_1/` with curriculum config and logging structure.
- [todo] Add `train_experiment_1.py` using functions from `rlvr_tiny/`.
- [todo] Run a smoke training pass with 1 epoch per phase and save logs/configs.
- [done] Simplify `train_experiment_1.py` into phase-oriented functions for config loading, preview, one-phase training, and full experiment run.
- [done] Rebuild the Colab notebook so config is editable in cells and each phase can be run or inspected separately.
- [done] Run a one-phase smoke check on the refactored path and confirm it still trains and logs.
- [done] Removed `arithmetic-transformer/verify_walkthrough.ipynb` as an unnecessary step after review. This was a wrong step and should not be restored unless explicitly needed.
- [done] Make `experiment_1/run_experiment_1.ipynb` fail loudly and show sanity output in Colab and local notebook runners.
- [done] Rebuild the notebook startup cells so local and Colab `Run all` use the same path setup flow.
- [done] Make notebook imports robust to stale kernels and cached old module versions.
