# Experiments

## LSTM 32K Toy Run: 1-digit vs 3-digit Addition on CPU

Environment:
- Python: `D:\Codex projects\.venv\Scripts\python.exe`
- Device: CPU
- Model: `lstm`
- Hidden size: `32`
- Layers: `4`
- Dropout: `0.01`
- Parameters: `34,831`

Shared command settings:
- `--train-batches 50`
- `--val-batches 20`
- `--batch-size 256`
- `--epochs 6`
- `--acc-next 1.1`

`--acc-next 1.1` was used to prevent automatic switching to longer numbers, so each run stayed fixed at one digit length.

### 1-digit addition

Command:

```powershell
& 'D:\Codex projects\.venv\Scripts\python.exe' train.py --kind lstm --device cpu --hidden-size 32 --num-layers 4 --dropout 0.01 --initial-number-length 1 --epochs 6 --train-batches 50 --val-batches 20 --batch-size 256 --acc-next 1.1
```

Validation accuracy by epoch:

| Epoch | Val acc |
| --- | --- |
| 1 | 0.010937 |
| 2 | 0.046289 |
| 3 | 0.21309 |
| 4 | 0.57227 |
| 5 | 0.86250 |
| 6 | 0.96328 |

Result:
- The 32K LSTM learned 1-digit addition well.
- By epoch 6 it reached `96.33%` validation accuracy.

Log:
- `runs/lstm_add_1digit_32k.txt`

### 3-digit addition

Command:

```powershell
& 'D:\Codex projects\.venv\Scripts\python.exe' train.py --kind lstm --device cpu --hidden-size 32 --num-layers 4 --dropout 0.01 --initial-number-length 3 --epochs 6 --train-batches 50 --val-batches 20 --batch-size 256 --acc-next 1.1
```

Validation accuracy by epoch:

| Epoch | Val acc |
| --- | --- |
| 1 | 0.0000000 |
| 2 | 0.0011719 |
| 3 | 0.0013672 |
| 4 | 0.00097656 |
| 5 | 0.0023438 |
| 6 | 0.0029297 |

Result:
- The same 32K LSTM did not learn 3-digit addition under this toy CPU budget.
- After 6 epochs it only reached `0.29%` validation accuracy.

Example wrong outputs near the end:
- `980 + 933 = 1913`, predicted `1111`
- `806 + 72 = 878`, predicted `1016`
- `244 + 695 = 939`, predicted `1011`

Log:
- `runs/lstm_add_3digit_32k_match1.txt`

## Conclusion

With the same ~32K LSTM and the same small CPU training budget:
- 1-digit addition is learnable quickly.
- 3-digit addition is not close to solved yet.

This toy result is consistent with the upstream README, where the 32K LSTM needs more training time as digit length increases.

## Notebook Checkpoint Validation

Checkpoint:
- `checkpoints/notebook_test/lstm_digit1_epoch1.pt`

Validation file:
- `runs/notebook_test/validation.json`

Purpose:
- verify that the notebook-style training path really writes a local checkpoint;
- verify that the checkpoint can be loaded and evaluated again.

Saved config in checkpoint:
- model: `lstm`
- operation: `add`
- hidden size: `32`
- layers: `4`
- heads: `4`
- dropout: `0.01`
- learning rate: `0.001`
- train batches: `1`
- validation batches during save run: `1`
- batch size during save run: `32`
- learned digit stage recorded: `1`
- saved accuracy in checkpoint metadata: `0.0`

Independent validation run:
- validation batches: `20`
- validation batch size: `256`
- device: `cpu`
- measured validation accuracy: `0.0`

Conclusion:
- the checkpoint save/load path works correctly;
- this particular checkpoint is only the forced 1-iteration notebook test checkpoint, not a learned model;
- it should be used as a pipeline validation artifact, not as a successful arithmetic result.
