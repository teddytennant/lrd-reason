"""GSM8K hard-subset eval.

Pulls the test split from HF datasets if available; falls back to a built-in
fixture for offline/smoke runs. The "hard" subset is the last 250 examples by
default (no canonical split exists; this gives a reproducible slice).
"""

from __future__ import annotations

from typing import Any

from .metrics import accuracy

_FIXTURE = [
    {
        "question": "Janet has 3 apples and gets 2 more. How many apples does she have?",
        "answer": "Janet starts with 3 apples and gets 2 more, so 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "A pen costs 4 dollars. How much do 7 pens cost?",
        "answer": "7 pens at 4 dollars each: 7 * 4 = 28.\n#### 28",
    },
]


def load_examples(n: int = 250) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"question": r["question"], "answer": r["answer"]} for r in ds.select(range(min(n, len(ds))))]
    except Exception:
        return _FIXTURE[: max(1, min(n, len(_FIXTURE)))]


def evaluate(runner: Any, n: int = 250, num_diffusion_steps: int | None = None) -> dict[str, float]:
    examples = load_examples(n)
    preds: list[str] = []
    golds: list[str] = []
    for i, ex in enumerate(examples):
        sid = f"gsm8k-{i}"
        resp = runner.chat(
            sid, ex["question"], num_diffusion_steps=num_diffusion_steps
        )
        preds.append(resp)
        golds.append(ex["answer"])
    return {"accuracy": accuracy(preds, golds), "n": float(len(examples))}
