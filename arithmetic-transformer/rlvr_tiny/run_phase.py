from copy import deepcopy
import csv
import json
from pathlib import Path
from typing import List, Optional, Union

import torch

from .dataset import build_phase_datasets
from .evaluate import evaluate_dataset, save_examples, save_metrics_table
from .format import FORMAT_A
from .train_rlvr import run_best_of_n, run_rlvr
from .train_sft import build_model, run_sft


def worked_decision(before: dict, after: dict, final_drop_tolerance: float = 0.01, process_gain: float = 0.03) -> str:
    final_drop = before["final_answer_accuracy"] - after["final_answer_accuracy"]
    process_improved = (
        after["parseable_trace_rate"] - before["parseable_trace_rate"] >= process_gain
        or after["full_step_valid_rate"] - before["full_step_valid_rate"] >= process_gain
    )
    if final_drop <= final_drop_tolerance and process_improved:
        return "worked"
    if final_drop <= final_drop_tolerance * 2:
        return "unclear"
    return "failed"


def _save_phase_checkpoint(model, out_dir: Path, label: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.pt"
    torch.save(model.state_dict(), path)
    return str(path)


def run_phase(phase, root_dir: Union[str, Path], baselines: Optional[List[str]] = None, device: str = "cpu"):
    root_dir = Path(root_dir)
    baselines = baselines or ["trace_sft_rlvr"]
    device_obj = torch.device(device)
    records = []
    for baseline in baselines:
        fmt_override = FORMAT_A if baseline == "answer_sft_only" else None
        datasets, spec = build_phase_datasets(phase, fmt_override=fmt_override)
        model = build_model(phase, spec, device_obj)
        phase_dir = root_dir / phase.name / baseline
        pre_metrics, pre_rows = evaluate_dataset(model, datasets["test"], spec, phase, device_obj, limit=32)
        save_metrics_table(pre_metrics, phase_dir / "pre_metrics.csv", phase_dir / "pre_metrics.md")
        save_examples(pre_rows, phase_dir / "pre_examples.json", limit=10)
        sft_info = run_sft(model, datasets["train"], spec, phase, device_obj)
        mid_metrics, mid_rows = evaluate_dataset(model, datasets["test"], spec, phase, device_obj, limit=32)
        save_metrics_table(mid_metrics, phase_dir / "post_sft_metrics.csv", phase_dir / "post_sft_metrics.md")
        save_examples(mid_rows, phase_dir / "post_sft_examples.json", limit=10)
        extra = {}
        post_metrics = mid_metrics
        post_rows = mid_rows
        if baseline == "trace_sft_rlvr" and phase.rl_enabled:
            extra = run_rlvr(model, datasets["train"], spec, phase, device_obj, out_dir=phase_dir)
            post_metrics, post_rows = evaluate_dataset(model, datasets["test"], spec, phase, device_obj, limit=32)
        elif baseline == "trace_sft_best_of_n" and phase.rl_enabled:
            extra = run_best_of_n(model, datasets["train"], spec, phase, device_obj, out_dir=phase_dir)
            post_metrics, post_rows = evaluate_dataset(model, datasets["test"], spec, phase, device_obj, limit=32)
        save_metrics_table(post_metrics, phase_dir / "final_metrics.csv", phase_dir / "final_metrics.md")
        save_examples(post_rows, phase_dir / "final_examples.json", limit=10)
        checkpoint_path = _save_phase_checkpoint(model, phase_dir, "phase_checkpoint")
        record = {
            "phase": phase.name,
            "baseline": baseline,
            "pre_metrics": pre_metrics,
            "post_sft_metrics": mid_metrics,
            "post_metrics": post_metrics,
            "decision": worked_decision(mid_metrics, post_metrics) if baseline != "answer_sft_only" else ("worked" if post_metrics["final_answer_accuracy"] >= 0.9 else "unclear"),
            "checkpoint": checkpoint_path,
            "sft_info": sft_info,
            **extra,
        }
        (phase_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    return records
