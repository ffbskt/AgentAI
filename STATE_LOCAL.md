Files:
- STATE_LOCAL.md
- AGENT.md
- PROTOCOL.md
- arithmetic-transformer/run_training.ipynb
- .gitignore
- arithmetic-transformer/README.md
- arithmetic-transformer/train.py
- arithmetic-transformer/requirements.txt
- arithmetic-transformer/requirements-colab.txt
- arithmetic-transformer/methods.py
- arithmetic-transformer/dataset.py
- arithmetic-transformer/model.py
- arithmetic-transformer/EXPERIMENTS.md
- arithmetic-transformer/runs/
- arithmetic-transformer/checkpoints/

Curent task [3]:
Prepare a slim Colab-ready workspace bundle with only required runnable files, add notebook support for Drive checkpoints, and document this structure in workspace instructions.

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
- [done] Initialize a root git repository for this workspace.
- [done] Add `.gitignore` rule for model checkpoints (`*.pt` only).
- [todo] Run the README-style 32K LSTM curriculum and save the 3-digit checkpoint.
- [todo] Update the experiment notes with the real run results.
