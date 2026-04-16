import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict

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


def load_config(config_path: Path, config_override: Dict[str, Any] = None):
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if config_override:
        payload = deep_update(payload, config_override)
    model_cfg = ModelConfig(**payload["model"])
    train_defaults = payload["train_defaults"]
    phases = []
    for phase in payload["phases"]:
        train_cfg = TrainConfig(**train_defaults)
        phases.append(
            PhaseConfig(
                name=phase["name"],
                description=phase["description"],
                fmt=phase["fmt"],
                shape=phase["shape"],
                carry_mode=phase["carry_mode"],
                rl_enabled=phase["rl_enabled"],
                model=deepcopy(model_cfg),
                train=train_cfg,
            )
        )
    return payload, phases


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def phase_summary_row(phase_name, post_sft_metrics, post_rlvr_metrics, checkpoint_path):
    return {
        "phase": phase_name,
        "post_sft_final_answer_accuracy": post_sft_metrics["final_answer_accuracy"],
        "post_sft_parseable_trace_rate": post_sft_metrics["parseable_trace_rate"],
        "post_rlvr_final_answer_accuracy": post_rlvr_metrics["final_answer_accuracy"],
        "post_rlvr_parseable_trace_rate": post_rlvr_metrics["parseable_trace_rate"],
        "post_rlvr_full_step_valid_rate": post_rlvr_metrics["full_step_valid_rate"],
        "checkpoint": str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiment_1/logs/smoke_run")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)
    config_override = json.loads(args.config_json) if args.config_json else None
    raw_config, phases = load_config(config_path, config_override=config_override)
    (root / "used_config.json").write_text(json.dumps(raw_config, indent=2), encoding="utf-8")

    device = torch.device(args.device)
    model = None
    summaries = []

    for phase in phases:
        datasets, spec = build_phase_datasets(phase)
        if model is None:
            model = build_model(phase, spec, device)
        else:
            model.ds = spec
            model = model.to(device)

        phase_dir = root / phase.name
        phase_dir.mkdir(parents=True, exist_ok=True)

        pre_metrics, pre_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=32)
        save_metrics_table(pre_metrics, phase_dir / "pre_metrics.csv", phase_dir / "pre_metrics.md")
        save_examples(pre_rows, phase_dir / "pre_examples.json", limit=10)

        sft_info = run_sft(model, datasets["train"], spec, phase, device)
        (phase_dir / "sft_info.json").write_text(json.dumps(sft_info, indent=2), encoding="utf-8")
        post_sft_metrics, post_sft_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=32)
        save_metrics_table(post_sft_metrics, phase_dir / "post_sft_metrics.csv", phase_dir / "post_sft_metrics.md")
        save_examples(post_sft_rows, phase_dir / "post_sft_examples.json", limit=10)

        rl_info = {}
        post_rlvr_metrics = post_sft_metrics
        post_rlvr_rows = post_sft_rows
        if phase.rl_enabled:
            rl_info = run_rlvr(model, datasets["train"], spec, phase, device, out_dir=phase_dir)
            (phase_dir / "rlvr_info.json").write_text(json.dumps(rl_info, indent=2), encoding="utf-8")
            post_rlvr_metrics, post_rlvr_rows = evaluate_dataset(model, datasets["test"], spec, phase, device, limit=32)

        save_metrics_table(post_rlvr_metrics, phase_dir / "post_rlvr_metrics.csv", phase_dir / "post_rlvr_metrics.md")
        save_examples(post_rlvr_rows, phase_dir / "post_rlvr_examples.json", limit=10)

        checkpoint_path = root.parent / "checkpoints" / f"{phase.name}.pt"
        save_checkpoint(model, checkpoint_path)

        summary = phase_summary_row(phase.name, post_sft_metrics, post_rlvr_metrics, checkpoint_path)
        summaries.append(summary)
        (phase_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
