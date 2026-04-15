import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .format import problem_string
from .verify import score_trace


def generate_completion(model, tokenizer, spec, prompt: str, device: torch.device, max_new_tokens: int, temperature: float = 1.0, sample: bool = False) -> str:
    prompt_ids = [tokenizer.start_token] + tokenizer.encode(prompt)
    seq = torch.full((spec.seq,), tokenizer.padding_token, dtype=torch.long, device=device)
    seq[: len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    current_len = len(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(seq[None])[0, current_len - 1]
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1).item() if sample else int(torch.argmax(probs))
        if token == tokenizer.eos_token_id:
            break
        generated.append(token)
        if current_len >= spec.seq:
            break
        seq[current_len] = token
        current_len += 1
    return tokenizer.decode(generated)


def evaluate_dataset(model, dataset, spec, phase, device: torch.device, limit: Optional[int] = None):
    tokenizer = spec.tokenizer
    model.eval()
    total = 0
    parse_rate = 0
    final_acc = 0
    full_valid = 0
    avg_valid_fraction = 0.0
    exact_trace = 0
    avg_len = 0.0
    rows = []
    count = min(len(dataset.prompts), limit) if limit is not None else len(dataset.prompts)
    with torch.no_grad():
        for idx in range(count):
            prompt = dataset.prompts[idx]
            completion = generate_completion(model, tokenizer, spec, prompt, device, phase.train.max_new_tokens)
            problem = prompt[:-1]
            scored = score_trace(problem, completion, reward_cfg=phase.reward, fmt=phase.fmt if phase.fmt != "E" else "C")
            total += 1
            parse_rate += int(scored["parse_ok"])
            final_acc += int(scored["final_ok"])
            full_valid += int(all(scored.get("step_valid_flags", [])) if scored["parse_ok"] else False)
            avg_valid_fraction += scored["valid_step_fraction"]
            exact_trace += int(scored.get("exact_trace_correct", False))
            avg_len += len(scored["full_trace"])
            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "full_trace": scored["full_trace"],
                    "parse_ok": scored["parse_ok"],
                    "final_ok": scored["final_ok"],
                    "valid_step_fraction": scored["valid_step_fraction"],
                    "reward_total": scored["reward_total"],
                    "error_type": scored["error_type"],
                }
            )
    metrics = {
        "final_answer_accuracy": final_acc / max(total, 1),
        "parseable_trace_rate": parse_rate / max(total, 1),
        "full_step_valid_rate": full_valid / max(total, 1),
        "average_valid_step_fraction": avg_valid_fraction / max(total, 1),
        "exact_full_trace_correct_rate": exact_trace / max(total, 1),
        "average_output_length": avg_len / max(total, 1),
        "num_examples": total,
    }
    return metrics, rows


def save_metrics_table(metrics: Dict, out_csv: Union[str, Path], out_md: Union[str, Path]) -> None:
    out_csv = Path(out_csv)
    out_md = Path(out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_examples(rows: List[Dict], out_path: Union[str, Path], limit: int = 10) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows[:limit], indent=2), encoding="utf-8")
