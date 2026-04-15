from copy import deepcopy
import json
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from .evaluate import generate_completion
from .verify import score_trace


def _sample_logprobs(model, spec, seqs: torch.Tensor, prompt_lengths: List[int]) -> torch.Tensor:
    logits = model(seqs)[:, :-1]
    targets = seqs[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    losses = []
    for i, prompt_len in enumerate(prompt_lengths):
        start = max(prompt_len - 1, 0)
        end = (targets[i] != spec.padding_token).nonzero()
        end_idx = int(end[-1].item()) + 1 if len(end) else start
        losses.append(gathered[i, start:end_idx].sum())
    return torch.stack(losses)


def _build_sequence(tokenizer, spec, prompt: str, completion: str, device: torch.device):
    full = [tokenizer.start_token] + tokenizer.encode(prompt + completion) + [tokenizer.eos_token_id]
    prompt_len = len([tokenizer.start_token] + tokenizer.encode(prompt))
    seq = full[: spec.seq] + [tokenizer.padding_token] * max(0, spec.seq - len(full))
    return torch.tensor(seq, dtype=torch.long, device=device), prompt_len


def run_best_of_n(model, dataset, spec, phase, device: torch.device, out_dir: Optional[Union[str, Path]] = None) -> dict:
    tokenizer = spec.tokenizer
    optimizer = model.configure_optimizers()
    logs = []
    for step in range(phase.train.best_of_n_steps):
        prompts = dataset.prompts[step * phase.train.rl_batch_size : (step + 1) * phase.train.rl_batch_size]
        selected = []
        for prompt in prompts:
            problem = prompt[:-1]
            candidates = []
            for _ in range(phase.train.num_generations):
                completion = generate_completion(model, tokenizer, spec, prompt, device, phase.train.max_new_tokens, temperature=phase.train.temperature, sample=True)
                scored = score_trace(problem, completion, reward_cfg=phase.reward, fmt=phase.fmt if phase.fmt != "E" else "C")
                candidates.append((scored["reward_total"], completion))
            best_reward, best_completion = max(candidates, key=lambda item: item[0])
            selected.append((prompt, best_completion, best_reward))
        if not selected:
            continue
        seqs = []
        for prompt, completion, _ in selected:
            seq, _ = _build_sequence(tokenizer, spec, prompt, completion, device)
            seqs.append(seq)
        batch = torch.stack(seqs)
        optimizer.zero_grad()
        logits = model(batch)[:, :-1]
        truth = batch[:, 1:]
        mask = truth != spec.padding_token
        loss = F.cross_entropy(logits[mask], truth[mask])
        loss.backward()
        optimizer.step()
        logs.append({"step": step, "loss": float(loss.item())})
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "best_of_n_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    return {"best_of_n_steps": len(logs)}


def run_rlvr(model, dataset, spec, phase, device: torch.device, out_dir: Optional[Union[str, Path]] = None) -> dict:
    tokenizer = spec.tokenizer
    reference = deepcopy(model).to(device)
    reference.eval()
    optimizer = model.configure_optimizers()
    logs = []
    prompts_all = dataset.prompts
    for step in range(phase.train.rl_steps):
        prompts = prompts_all[step * phase.train.rl_batch_size : (step + 1) * phase.train.rl_batch_size]
        if not prompts:
            break
        sampled_seqs = []
        prompt_lengths = []
        rewards = []
        for prompt in prompts:
            problem = prompt[:-1]
            group = []
            seq_group = []
            len_group = []
            for _ in range(phase.train.num_generations):
                completion = generate_completion(model, tokenizer, spec, prompt, device, phase.train.max_new_tokens, temperature=phase.train.temperature, sample=True)
                scored = score_trace(problem, completion, reward_cfg=phase.reward, fmt=phase.fmt if phase.fmt != "E" else "C")
                seq, prompt_len = _build_sequence(tokenizer, spec, prompt, completion, device)
                group.append(scored["reward_total"])
                seq_group.append(seq)
                len_group.append(prompt_len)
            group_tensor = torch.tensor(group, dtype=torch.float32, device=device)
            advantages = (group_tensor - group_tensor.mean()) / (group_tensor.std() + 1e-6)
            rewards.extend(advantages.tolist())
            sampled_seqs.extend(seq_group)
            prompt_lengths.extend(len_group)
        seq_batch = torch.stack(sampled_seqs)
        advantage_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        optimizer.zero_grad()
        seq_logprob = _sample_logprobs(model, spec, seq_batch, prompt_lengths)
        with torch.no_grad():
            ref_logprob = _sample_logprobs(reference, spec, seq_batch, prompt_lengths)
        kl_term = ((seq_logprob - ref_logprob) ** 2).mean()
        loss = -(advantage_tensor * seq_logprob).mean() + phase.train.kl_coef * kl_term
        loss.backward()
        optimizer.step()
        logs.append({"step": step, "loss": float(loss.item()), "mean_advantage": float(advantage_tensor.mean().item()), "kl_term": float(kl_term.item())})
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "rlvr_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    return {"rlvr_steps": len(logs)}
