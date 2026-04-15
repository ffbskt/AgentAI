from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional, Union

from .format import FORMAT_A, FORMAT_B, FORMAT_C, FORMAT_D
from .verify import RewardConfig


@dataclass
class ModelConfig:
    kind: str = "transformer-sine"
    hidden_size: int = 64
    ffw_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    lr: float = 1e-3


@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 64
    train_samples: int = 256
    val_samples: int = 128
    test_samples: int = 128
    max_new_tokens: int = 32
    sft_only: bool = False
    rl_steps: int = 8
    rl_batch_size: int = 16
    num_generations: int = 4
    temperature: float = 1.0
    kl_coef: float = 0.02
    best_of_n_steps: int = 4
    seed: int = 7


@dataclass
class PhaseConfig:
    name: str
    description: str
    fmt: str
    shape: str
    carry_mode: str
    rl_enabled: bool
    sft_warmup: bool = True
    mixed_formats: Optional[dict] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


DEFAULT_PHASES = [
    PhaseConfig("phase0_baseline", "Baseline check on current capability", FORMAT_A, "3d+3d", "any", False, train=TrainConfig(epochs=1, train_samples=64, val_samples=64, test_samples=64)),
    PhaseConfig("phase1_final_1d1d", "1-digit final answer warmup", FORMAT_A, "1d+1d", "any", False, train=TrainConfig(epochs=2, train_samples=256, val_samples=128, test_samples=128)),
    PhaseConfig("phase2_trace_2d1d", "2-digit plus 1-digit short traces", FORMAT_B, "2d+1d", "easy", True, train=TrainConfig(epochs=2, train_samples=384, val_samples=128, test_samples=128, rl_steps=10, best_of_n_steps=6)),
    PhaseConfig("phase3_trace_2d2d_simple", "2-digit plus 2-digit moderate traces", FORMAT_C, "2d+2d", "simple", True, train=TrainConfig(epochs=2, train_samples=384, val_samples=128, test_samples=128, rl_steps=10)),
    PhaseConfig("phase4_trace_2d2d_carry", "2-digit plus 2-digit carry cases", FORMAT_C, "2d+2d", "heavy", True, train=TrainConfig(epochs=2, train_samples=384, val_samples=128, test_samples=128, rl_steps=12)),
    PhaseConfig("phase5_trace_3d_mix", "3-digit extension", FORMAT_C, "3d+3d", "any", True, train=TrainConfig(epochs=2, train_samples=512, val_samples=128, test_samples=128, rl_steps=12, max_new_tokens=48)),
    PhaseConfig("phase6_compressed_carry", "Compressed carry notation", FORMAT_D, "2d+2d", "heavy", True, train=TrainConfig(epochs=2, train_samples=384, val_samples=128, test_samples=128, rl_steps=10)),
    PhaseConfig("phase7_mixed", "Mixed robust training", "E", "3d+3d", "any", True, mixed_formats={"A": 0.3, "B": 0.2, "C": 0.5}, train=TrainConfig(epochs=2, train_samples=512, val_samples=128, test_samples=128, rl_steps=12, max_new_tokens=48)),
]


def get_phase(name: str) -> PhaseConfig:
    for phase in DEFAULT_PHASES:
        if phase.name == name:
            return phase
    raise KeyError(name)


def export_default_curriculum_json(out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for phase in DEFAULT_PHASES:
        record = asdict(phase)
        payload.append(record)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
