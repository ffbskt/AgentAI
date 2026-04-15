from dataclasses import dataclass
import random
from typing import List, Optional, Tuple


FORMAT_A = "A"
FORMAT_B = "B"
FORMAT_C = "C"
FORMAT_D = "D"
FORMAT_E = "E"


def final_answer_from_trace(trace: str) -> str:
    return trace.rsplit("=", 1)[-1]


def problem_string(a: int, b: int) -> str:
    return f"{a}+{b}"


def decompose_number(n: int) -> List[int]:
    if n == 0:
        return [0]
    terms = []
    value = n
    place = 1
    while value > 0:
        digit = value % 10
        if digit:
            terms.append(digit * place)
        value //= 10
        place *= 10
    return list(reversed(terms))


def _group_decomposed_terms(terms: List[int]) -> List[int]:
    groups: dict[int, int] = {}
    for term in terms:
        if term == 0:
            groups[0] = groups.get(0, 0) + term
            continue
        magnitude = 1
        while term % (magnitude * 10) == 0:
            magnitude *= 10
        groups[magnitude] = groups.get(magnitude, 0) + term
    ordered = [groups[k] for k in sorted(groups.keys(), reverse=True) if groups[k] != 0]
    return ordered or [0]


def expr_from_terms(terms: List[int]) -> str:
    return "+".join(str(t) for t in terms)


def compressed_carry_trace(a: int, b: int) -> str:
    columns = []
    carry = 0
    aa = list(map(int, str(a)[::-1]))
    bb = list(map(int, str(b)[::-1]))
    max_len = max(len(aa), len(bb))
    for i in range(max_len):
        da = aa[i] if i < len(aa) else 0
        db = bb[i] if i < len(bb) else 0
        total = da + db + carry
        columns.append(str(total))
        carry = total // 10
    columns.append(str(carry))
    return "|".join(columns) + f"={a+b}"


def canonical_full_trace(a: int, b: int, fmt: str) -> str:
    prompt = problem_string(a, b)
    if fmt == FORMAT_A:
        return f"{prompt}={a+b}"
    if fmt in (FORMAT_B, FORMAT_C):
        terms = decompose_number(a) + decompose_number(b)
        grouped = _group_decomposed_terms(terms)
        return f"{prompt}={expr_from_terms(terms)}={expr_from_terms(grouped)}={a+b}"
    if fmt == FORMAT_D:
        return f"{prompt}={compressed_carry_trace(a, b)}"
    raise ValueError(f"Unsupported format {fmt}")


def canonical_target(a: int, b: int, fmt: str) -> str:
    return canonical_full_trace(a, b, fmt).split("=", 1)[1]


@dataclass
class MixedFormatConfig:
    formats: Tuple[str, ...]
    probs: Tuple[float, ...]

    def sample(self, rng: random.Random) -> str:
        threshold = rng.random()
        total = 0.0
        for fmt, prob in zip(self.formats, self.probs):
            total += prob
            if threshold <= total:
                return fmt
        return self.formats[-1]


def resolve_format(fmt: str, rng: random.Random, mixed: Optional[MixedFormatConfig] = None) -> str:
    if fmt != FORMAT_E:
        return fmt
    if mixed is None:
        raise ValueError("FORMAT_E requires mixed config")
    return mixed.sample(rng)
