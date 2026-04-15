from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AdditionModel

from .dataset import phase_args_from_config


def answer_mask(spec, batch):
    mask = torch.cumsum(batch == spec.end_token, dim=1) == 1
    mask &= batch != spec.end_token
    return mask[:, 1:]


def trace_loss(model, spec, batch):
    mask = answer_mask(spec, batch)
    truth = batch[:, 1:]
    out = model(batch)[:, :-1]
    return F.cross_entropy(out[mask], truth[mask])


def build_model(phase, spec, device: torch.device):
    model = AdditionModel(
        ds=spec,
        kind=phase.model.kind,
        hidden_size=phase.model.hidden_size,
        ffw_size=phase.model.ffw_size,
        num_layers=phase.model.num_layers,
        num_heads=phase.model.num_heads,
        lr=phase.model.lr,
        dropout=phase.model.dropout,
    )
    return model.to(device)


def clone_model(model, device: torch.device):
    cloned = deepcopy(model)
    return cloned.to(device)


def run_sft(model, dataset, spec, phase, device: torch.device) -> dict:
    model.train()
    loader = DataLoader(dataset, batch_size=phase.train.batch_size, shuffle=True)
    optimizer = model.configure_optimizers()
    epoch_losses = []
    for _ in range(phase.train.epochs):
        batch_losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = trace_loss(model, spec, batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses) / max(len(batch_losses), 1))
    return {"sft_loss_history": epoch_losses}
