from dataclasses import dataclass
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from .curriculum import PhaseConfig
from .format import FORMAT_E, MixedFormatConfig, canonical_full_trace, canonical_target, problem_string, resolve_format


def split_shape(shape: str) -> Tuple[int, int]:
    left, right = shape.split("+")
    return int(left[0]), int(right[0])


def count_carries(a: int, b: int) -> int:
    carries = 0
    carry = 0
    aa = list(map(int, str(a)[::-1]))
    bb = list(map(int, str(b)[::-1]))
    for i in range(max(len(aa), len(bb))):
        da = aa[i] if i < len(aa) else 0
        db = bb[i] if i < len(bb) else 0
        total = da + db + carry
        if total >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
    return carries


def _fits_carry_mode(a: int, b: int, mode: str) -> bool:
    carries = count_carries(a, b)
    if mode == "any":
        return True
    if mode in ("no", "no_carry"):
        return carries == 0
    if mode in ("easy", "simple"):
        return carries <= 1
    if mode == "heavy":
        return carries >= 1
    return True


def _sample_number(digits: int, rng: random.Random) -> int:
    low = 0 if digits == 1 else 10 ** (digits - 1)
    high = 10**digits - 1
    return rng.randint(low, high)


def generate_problem(shape: str, carry_mode: str, rng: random.Random) -> Tuple[int, int]:
    da, db = split_shape(shape)
    for _ in range(1000):
        a = _sample_number(da, rng)
        b = _sample_number(db, rng)
        if _fits_carry_mode(a, b, carry_mode):
            return a, b
    raise RuntimeError(f"Could not sample problem for {shape=} {carry_mode=}")


class TraceTokenizer:
    def __init__(self):
        chars = list("0123456789+=|")
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.itos = [self.pad_token, self.bos_token, self.eos_token] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.padding_token = self.stoi[self.pad_token]
        self.start_token = self.stoi[self.bos_token]
        self.eos_token_id = self.stoi[self.eos_token]
        self.end_token = self.stoi["="]
        self.n_tokens = len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Union[List[int], torch.Tensor], strip_special: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for idx in ids:
            token = self.itos[idx]
            if strip_special and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            chars.append(token)
        return "".join(chars)


@dataclass
class TraceSpec:
    tokenizer: TraceTokenizer
    seq: int

    @property
    def n_tokens(self) -> int:
        return self.tokenizer.n_tokens

    @property
    def padding_token(self) -> int:
        return self.tokenizer.padding_token

    @property
    def end_token(self) -> int:
        return self.tokenizer.end_token

    def repr_example(self, example: Union[torch.Tensor, List[int]]) -> str:
        return self.tokenizer.decode(example)


class SymbolicAdditionDataset(Dataset):
    def __init__(self, sequences: List[List[int]], prompts: List[str], targets: List[str], traces: List[str]):
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.prompts = prompts
        self.targets = targets
        self.traces = traces

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def _max_trace_length(phase: PhaseConfig, fmt: str) -> int:
    da, db = split_shape(phase.shape)
    a = 10**da - 1
    b = 10**db - 1
    return len(canonical_full_trace(a, b, fmt))


def _build_sequences(phase: PhaseConfig, count: int, seed: int, fmt_override: Optional[str] = None) -> Tuple[SymbolicAdditionDataset, TraceSpec]:
    rng = random.Random(seed)
    tokenizer = TraceTokenizer()
    mixed = None
    fmt_choice = fmt_override or phase.fmt
    if fmt_choice == FORMAT_E:
        mixed = MixedFormatConfig(tuple(phase.mixed_formats.keys()), tuple(phase.mixed_formats.values()))
        max_fmt_len = max(_max_trace_length(phase, item_fmt) for item_fmt in phase.mixed_formats.keys())
    else:
        max_fmt_len = _max_trace_length(phase, fmt_choice)
    seq_len = max_fmt_len + 3
    spec = TraceSpec(tokenizer=tokenizer, seq=seq_len)

    sequences, prompts, targets, traces = [], [], [], []
    for _ in range(count):
        a, b = generate_problem(phase.shape, phase.carry_mode, rng)
        resolved_fmt = resolve_format(fmt_choice, rng, mixed)
        prompt = problem_string(a, b) + "="
        target = canonical_target(a, b, resolved_fmt)
        full = prompt + target
        encoded = [tokenizer.start_token] + tokenizer.encode(full) + [tokenizer.eos_token_id]
        encoded += [tokenizer.padding_token] * (seq_len - len(encoded))
        sequences.append(encoded[:seq_len])
        prompts.append(prompt)
        targets.append(target)
        traces.append(full)
    return SymbolicAdditionDataset(sequences, prompts, targets, traces), spec


def edge_case_problems() -> List[Tuple[int, int]]:
    return [(9, 9), (19, 9), (99, 1), (58, 67), (199, 801), (123, 458)]


def build_phase_datasets(phase: PhaseConfig, fmt_override: Optional[str] = None) -> Tuple[Dict, TraceSpec]:
    train_ds, spec = _build_sequences(phase, phase.train.train_samples, phase.train.seed, fmt_override)
    val_ds, _ = _build_sequences(phase, phase.train.val_samples, phase.train.seed + 1, fmt_override)
    test_ds, _ = _build_sequences(phase, phase.train.test_samples, phase.train.seed + 2, fmt_override)
    return {"train": train_ds, "val": val_ds, "test": test_ds, "edge_cases": edge_case_problems()}, spec


def phase_args_from_config(phase: PhaseConfig, spec: TraceSpec) -> SimpleNamespace:
    return SimpleNamespace(
        preferred_dtype="int64",
        base=10,
        number_length=max(split_shape(phase.shape)),
        pre_end_padding=0,
        flip=False,
        kind=phase.model.kind,
        hidden_size=phase.model.hidden_size,
        ffw_size=phase.model.ffw_size,
        num_layers=phase.model.num_layers,
        num_heads=phase.model.num_heads,
        lr=phase.model.lr,
        dropout=phase.model.dropout,
        op="add",
        initial_number_length=1,
        device="cpu",
        seq=spec.seq,
    )
