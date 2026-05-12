"""MATH benchmark eval (Hendrycks et al.)."""

from __future__ import annotations

from typing import Any

from .metrics import accuracy

_FIXTURE = [
    {
        "problem": "What is 12 * 11?",
        "solution": "12 * 11 = 132. \\boxed{132}",
    },
    {
        "problem": "Solve x + 5 = 12.",
        "solution": "x = 12 - 5 = 7. \\boxed{7}",
    },
]


def load_examples(n: int = 250) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("hendrycks/competition_math", split="test")
        return [
            {"problem": r["problem"], "solution": r["solution"]}
            for r in ds.select(range(min(n, len(ds))))
        ]
    except Exception:
        return _FIXTURE[: max(1, min(n, len(_FIXTURE)))]


def evaluate(runner: Any, n: int = 250, num_diffusion_steps: int | None = None) -> dict[str, float]:
    examples = load_examples(n)
    preds: list[str] = []
    golds: list[str] = []
    for i, ex in enumerate(examples):
        sid = f"math-{i}"
        resp = runner.chat(sid, ex["problem"], num_diffusion_steps=num_diffusion_steps)
        preds.append(resp)
        golds.append(ex["solution"])
    return {"accuracy": accuracy(preds, golds), "n": float(len(examples))}
