import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

from .curriculum import DEFAULT_PHASES
from .run_phase import run_phase


ACTIONABLE_SUMMARY = """\
## Literature Summary

- DeepSeekMath motivates the order used here: first build base competence, then supervised reasoning traces, then reinforcement learning with verifiable rewards.
- Process supervision matters because final-answer-only reward can still permit shortcutting or malformed intermediate traces.
- Group-relative policy updates are a reasonable fit for small symbolic tasks because they compare several sampled trajectories for the same prompt.
- KL-style anchoring or a reference-policy penalty helps protect base competence during RL-style updates.
- Short validation runs before scaling up are important; this framework keeps each phase compact and inspectable before any larger run.

Sources:
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- Hugging Face TRL GRPO docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
"""


def _phase_baselines(phase_name: str):
    if phase_name == "phase2_trace_2d1d":
        return ["answer_sft_only", "trace_sft_only", "trace_sft_rlvr", "trace_sft_best_of_n"]
    if phase_name.startswith("phase") and phase_name != "phase0_baseline":
        return ["trace_sft_only", "trace_sft_rlvr"]
    return ["answer_sft_only"]


def write_task_results(root_dir: Path, summary_records: List[Dict]) -> None:
    out_path = root_dir.parent.parent / "TASK_RESULTS.md"
    lines = [ACTIONABLE_SUMMARY, "", "# Phase Results", ""]
    for record in summary_records:
        lines.append(f"## {record['phase']} / {record['baseline']}")
        lines.append(f"- decision: `{record['decision']}`")
        lines.append(f"- checkpoint: `{record['checkpoint']}`")
        lines.append(f"- pre final-answer acc: `{record['pre_metrics']['final_answer_accuracy']:.4f}`")
        lines.append(f"- post SFT final-answer acc: `{record['post_sft_metrics']['final_answer_accuracy']:.4f}`")
        lines.append(f"- final-answer acc after final step: `{record['post_metrics']['final_answer_accuracy']:.4f}`")
        lines.append(f"- parseable trace rate after final step: `{record['post_metrics']['parseable_trace_rate']:.4f}`")
        lines.append(f"- full-step-valid rate after final step: `{record['post_metrics']['full_step_valid_rate']:.4f}`")
        lines.append(f"- average valid-step fraction after final step: `{record['post_metrics']['average_valid_step_fraction']:.4f}`")
        lines.append(f"- average output length after final step: `{record['post_metrics']['average_output_length']:.2f}`")
        if "rlvr_steps" in record:
            lines.append(f"- RLVR steps: `{record['rlvr_steps']}`")
        if "best_of_n_steps" in record:
            lines.append(f"- best-of-N steps: `{record['best_of_n_steps']}`")
        lines.append("")
    recommendation = "Keep phase ordering curriculum-first, prefer trace SFT before RLVR, and compare against best-of-N on nontrivial phases because small models may benefit almost as much from self-training as from policy optimization."
    lines.append("## Recommendation")
    lines.append(recommendation)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/rlvr_tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase-limit", type=int, default=3)
    args = parser.parse_args()

    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    summary_records = []
    for phase in DEFAULT_PHASES[: args.phase_limit]:
        baselines = _phase_baselines(phase.name)
        summary_records.extend(run_phase(phase, root_dir, baselines=baselines, device=args.device))
    csv_path = root_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "baseline", "decision", "final_answer_accuracy", "parseable_trace_rate", "full_step_valid_rate", "checkpoint"])
        for record in summary_records:
            writer.writerow([
                record["phase"],
                record["baseline"],
                record["decision"],
                record["post_metrics"]["final_answer_accuracy"],
                record["post_metrics"]["parseable_trace_rate"],
                record["post_metrics"]["full_step_valid_rate"],
                record["checkpoint"],
            ])
    (root_dir / "summary.json").write_text(json.dumps(summary_records, indent=2), encoding="utf-8")
    write_task_results(root_dir, summary_records)


if __name__ == "__main__":
    main()
