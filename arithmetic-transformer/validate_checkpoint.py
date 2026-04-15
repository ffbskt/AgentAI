import argparse
import json
from types import SimpleNamespace

import torch

from model import AdditionModel
from train import make_dataset, validation_step


def load_args(ckpt_args):
    return SimpleNamespace(**ckpt_args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = load_args(checkpoint["args"])
    dataset = make_dataset(ckpt_args, number_length=checkpoint["number_length"])

    model = AdditionModel(
        ds=dataset,
        kind=ckpt_args.kind,
        hidden_size=ckpt_args.hidden_size,
        ffw_size=2 * ckpt_args.hidden_size
        if ckpt_args.ffw_size is None
        else ckpt_args.ffw_size,
        num_layers=ckpt_args.num_layers,
        num_heads=ckpt_args.num_heads,
        lr=ckpt_args.lr,
        dropout=ckpt_args.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    accs = []
    with torch.no_grad():
        np_data = dataset.generate_batch(args.batch_size * args.val_batches)
        val_data = torch.tensor(np_data).to(device)
        for batch_idx in range(args.val_batches):
            batch = val_data[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
            accs.append(validation_step(model, batch).item())

    result = {
        "checkpoint": args.checkpoint,
        "validated_number_length": checkpoint["number_length"],
        "saved_accuracy": checkpoint.get("accuracy"),
        "measured_accuracy": sum(accs) / len(accs),
        "epochs_per_digit": checkpoint.get("epochs_per_digit"),
        "config": checkpoint.get("config"),
        "validation_batches": args.val_batches,
        "validation_batch_size": args.batch_size,
        "device": args.device,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
