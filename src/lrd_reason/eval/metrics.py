"""Scalar metrics shared across eval modules."""

from __future__ import annotations

import re
import statistics
import time
from collections.abc import Callable
from typing import Any

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Last numeric literal in the string. Matches GSM8K/MATH gold format."""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    correct = 0
    for p, g in zip(preds, golds, strict=True):
        p_num = extract_final_number(p)
        g_num = extract_final_number(g) or g.strip()
        if p_num is not None and p_num == g_num:
            correct += 1
    return correct / len(preds)


def contradiction_rate(responses: list[str], judge: Callable[[str, str], bool]) -> float:
    """Fraction of consecutive response pairs flagged as contradictory by `judge`."""
    if len(responses) < 2:
        return 0.0
    contradictions = sum(
        1 for a, b in zip(responses[:-1], responses[1:], strict=True) if judge(a, b)
    )
    return contradictions / (len(responses) - 1)


def latency_p50_p95(fn: Callable[[], Any], n_warmup: int = 1, n_iters: int = 20) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "p50": statistics.median(times),
        "p95": times[int(0.95 * (len(times) - 1))],
        "mean": statistics.mean(times),
    }
