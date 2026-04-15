from dataclasses import dataclass
from typing import List, Optional, Tuple

from .format import FORMAT_D, final_answer_from_trace


ALLOWED_CHARS = set("0123456789+=|")


@dataclass
class RewardConfig:
    w_syntax: float = 0.2
    w_steps: float = 0.3
    w_final: float = 0.6
    w_length: float = 0.01
    malformed_penalty: float = -0.5


def parse_trace(s: str) -> Tuple[List[str], bool, Optional[str]]:
    if not s:
        return [], False, "empty"
    if any(ch not in ALLOWED_CHARS for ch in s):
        return [], False, "bad_char"
    parts = s.split("=")
    if any(part == "" for part in parts):
        return parts, False, "empty_segment"
    return parts, True, None


def eval_expr(expr: str) -> int:
    if expr == "":
        raise ValueError("empty_expr")
    if expr.startswith("+") or expr.endswith("+") or "++" in expr:
        raise ValueError("bad_plus")
    parts = expr.split("+")
    if any(not token.isdigit() for token in parts):
        raise ValueError("non_digit")
    return sum(int(token) for token in parts)


def _parse_problem(problem: str) -> Tuple[int, int]:
    left, right = problem.split("+")
    return int(left), int(right)


def _check_compressed(problem: str, packed: str) -> bool:
    if "|" not in packed:
        return False
    a, b = _parse_problem(problem)
    columns = packed.split("|")
    carry = 0
    aa = list(map(int, str(a)[::-1]))
    bb = list(map(int, str(b)[::-1]))
    max_len = max(len(aa), len(bb))
    expected = []
    for i in range(max_len):
        da = aa[i] if i < len(aa) else 0
        db = bb[i] if i < len(bb) else 0
        total = da + db + carry
        expected.append(str(total))
        carry = total // 10
    expected.append(str(carry))
    return columns == expected


def check_local_steps(parts: List[str], fmt: str = "A") -> Tuple[List[bool], float]:
    if len(parts) < 2:
        return [], 0.0
    valids: List[bool] = []
    if fmt == FORMAT_D:
        problem = parts[0]
        if len(parts) >= 2:
            valids.append(_check_compressed(problem, parts[1]))
        if len(parts) >= 3:
            try:
                valids.append(int(parts[-1]) == eval_expr(problem))
            except ValueError:
                valids.append(False)
        frac = sum(valids) / len(valids) if valids else 0.0
        return valids, frac
    for left, right in zip(parts[:-1], parts[1:]):
        try:
            valids.append(eval_expr(left) == eval_expr(right))
        except ValueError:
            valids.append(False)
    frac = sum(valids) / len(valids) if valids else 0.0
    return valids, frac


def extract_final_answer(parts: List[str]) -> str:
    return parts[-1]


def score_trace(problem: str, trace: str, reward_cfg: Optional[RewardConfig] = None, fmt: str = "A") -> dict:
    reward_cfg = reward_cfg or RewardConfig()
    full_trace = trace if trace.startswith(problem + "=") else f"{problem}={trace}"
    parts, parse_ok, error = parse_trace(full_trace)
    if not parse_ok:
        return {
            "parse_ok": False,
            "final_ok": False,
            "num_steps": 0,
            "valid_step_fraction": 0.0,
            "reward_total": reward_cfg.malformed_penalty,
            "error_type": error,
            "full_trace": full_trace,
            "final_answer": final_answer_from_trace(full_trace),
            "step_valid_flags": [],
        }
    step_valid_flags, valid_step_fraction = check_local_steps(parts, fmt=fmt)
    try:
        final_ok = int(extract_final_answer(parts)) == eval_expr(problem)
    except ValueError:
        final_ok = False
        error = "bad_final"
    length_penalty = len(full_trace)
    reward = (
        reward_cfg.w_syntax * 1.0
        + reward_cfg.w_steps * valid_step_fraction
        + reward_cfg.w_final * float(final_ok)
        - reward_cfg.w_length * length_penalty / 100.0
    )
    return {
        "parse_ok": True,
        "final_ok": final_ok,
        "num_steps": max(0, len(parts) - 1),
        "valid_step_fraction": valid_step_fraction,
        "reward_total": reward,
        "error_type": error,
        "full_trace": full_trace,
        "final_answer": extract_final_answer(parts),
        "step_valid_flags": step_valid_flags,
        "exact_trace_correct": final_ok and all(step_valid_flags),
    }
