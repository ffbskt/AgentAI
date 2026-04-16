import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rlvr_tiny.curriculum import ModelConfig, PhaseConfig, TrainConfig
from rlvr_tiny.dataset import build_phase_datasets
from rlvr_tiny.evaluate import evaluate_dataset, save_examples, save_metrics_table
from rlvr_tiny.train_rlvr import run_rlvr
from rlvr_tiny.train_sft import build_model, run_sft


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_experiment_config(config_path: str = "config.json", config_json: Optional[str] = None) -> Tuple[Dict[str, Any], List[PhaseConfig]]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config_json:
        payload = deep_update(payload, json.loads(config_json))

    model_cfg = ModelConfig(**payload["model"])
    train_defaults = payload["train_defaults"]
    phases: List[PhaseConfig] = []
    for phase in payload["phases"]:
        phases.append(
            PhaseConfig(
                name=phase["name"],
                description=phase["description"],
                fmt=phase["fmt"],
                shape=phase["shape"],
                carry_mode=phase["carry_mode"],
                rl_enabled=phase["rl_enabled"],
                model=deepcopy(model_cfg),
                train=TrainConfig(**train_defaults),
            )
        )
    return payload, phases


def preview_phase_samples(phase: PhaseConfig, sample_count: int = 5) -> List[Dict[str, str]]:
    datasets, _ = build_phase_datasets(phase)
    preview = []
    for idx in range(min(sample_count, len(datasets["train"].traces))):
        preview.append(
            {
                "prompt": datasets["train"].prompts[idx],
                "target": datasets["train"].targets[idx],
                "full_trace": datasets["train"].traces[idx],
            }
        )
    return preview


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def run_one_phase(
    phase: PhaseConfig,
    model,
    device: torch.device,
    output_root: Path,
    sample_limit: int = 32,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    datasets, spec = build_phase_datasets(phase)
    if model is None:
        model = build_model(phase, spec, device)
    else:
        model.ds = spec
        model = model.to(device)

    phase_dir = output_root / phase.name
    phase_dir.mkdir(parents=True, exist_ok=True)

    pre_metrics, pre_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=sample_limit)
    save_metrics_table(pre_metrics, phase_dir / "pre_metrics.csv", phase_dir / "pre_metrics.md")
    save_examples(pre_rows, phase_dir / "pre_examples.json", limit=10)

    sft_info = run_sft(model, datasets["train"], spec, phase, device)
    _write_json(phase_dir / "sft_info.json", sft_info)
    post_sft_metrics, post_sft_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=sample_limit)
    save_metrics_table(post_sft_metrics, phase_dir / "post_sft_metrics.csv", phase_dir / "post_sft_metrics.md")
    save_examples(post_sft_rows, phase_dir / "post_sft_examples.json", limit=10)

    rlvr_info = {}
    post_rlvr_metrics = post_sft_metrics
    post_rlvr_rows = post_sft_rows
    if phase.rl_enabled:
        rlvr_info = run_rlvr(model, datasets["train"], spec, phase, device, out_dir=phase_dir)
        _write_json(phase_dir / "rlvr_info.json", rlvr_info)
        post_rlvr_metrics, post_rlvr_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=sample_limit)

    save_metrics_table(post_rlvr_metrics, phase_dir / "post_rlvr_metrics.csv", phase_dir / "post_rlvr_metrics.md")
    save_examples(post_rlvr_rows, phase_dir / "post_rlvr_examples.json", limit=10)

    checkpoint_path = output_root.parent / "checkpoints" / f"{phase.name}.pt"
    _save_checkpoint(model, checkpoint_path)

    summary = {
        "phase": phase.name,
        "description": phase.description,
        "pre_metrics": pre_metrics,
        "post_sft_metrics": post_sft_metrics,
        "post_rlvr_metrics": post_rlvr_metrics,
        "sft_info": sft_info,
        "rlvr_info": rlvr_info,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(phase_dir / "summary.json", summary)
    return model, summary


def run_experiment(
    config_path: str = "config.json",
    config_json: Optional[str] = None,
    device: str = "cpu",
    output_dir: str = "logs/smoke_run",
    phase_name: Optional[str] = None,
    phase_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    raw_config, phases = load_experiment_config(config_path=config_path, config_json=config_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "used_config.json", raw_config)

    if phase_name is not None:
        phases = [phase for phase in phases if phase.name == phase_name]
    elif phase_index is not None:
        phases = [phases[phase_index]]

    device_obj = torch.device(device)
    model = None
    summaries: List[Dict[str, Any]] = []
    for phase in phases:
        model, summary = run_one_phase(phase, model, device_obj, output_root)
        summaries.append(
            {
                "phase": summary["phase"],
                "post_sft_final_answer_accuracy": summary["post_sft_metrics"]["final_answer_accuracy"],
                "post_sft_parseable_trace_rate": summary["post_sft_metrics"]["parseable_trace_rate"],
                "post_rlvr_final_answer_accuracy": summary["post_rlvr_metrics"]["final_answer_accuracy"],
                "post_rlvr_parseable_trace_rate": summary["post_rlvr_metrics"]["parseable_trace_rate"],
                "post_rlvr_full_step_valid_rate": summary["post_rlvr_metrics"]["full_step_valid_rate"],
                "checkpoint": summary["checkpoint"],
            }
        )

    if summaries:
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        _write_json(output_root / "summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="logs/smoke_run")
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--phase-index", type=int, default=None)
    args = parser.parse_args()

    run_experiment(
        config_path=args.config,
        config_json=args.config_json,
        device=args.device,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        phase_index=args.phase_index,
    )


if __name__ == "__main__":
    main()
