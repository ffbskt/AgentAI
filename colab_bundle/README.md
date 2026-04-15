# Colab Bundle

This folder is the slim runtime bundle for Google Colab or a small git checkout.

Included files:
- `train.py`
- `dataset.py`
- `model.py`
- `methods.py`
- `requirements-colab.txt`
- `run_training_colab.ipynb`

Use this folder when you want to upload or clone only the minimum runnable files.

Expected workflow:
1. Open `run_training_colab.ipynb` in Colab.
2. Mount Google Drive.
3. Set the path to this folder in Drive or in `/content`.
4. Run training from this bundle.
5. Save checkpoints and protocol logs to Drive.

Recommended outputs:
- checkpoints in `.../checkpoints/`
- protocol log in `.../runs/protocol.jsonl`
- stdout log in `.../runs/train.log`
