"""Multi-turn consistency + self-correction eval.

Hand-crafted scenarios as in-file fixtures. Each scenario is a list of turns; the
metric is whether the model's later answers contradict its earlier ones (judged
heuristically by token overlap on the answer span, or by a callable judge if one
is supplied).

Self-correction: inject a deliberately wrong premise as turn 0 and check whether
the model recovers when given new evidence in turn 1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "fact-stability",
        "turns": [
            "Alice has 3 apples.",
            "Bob gives Alice 2 more apples. How many apples does Alice have now?",
            "Wait, how many apples did Alice start with?",
        ],
        "expected_keywords_per_turn": [[], ["5"], ["3"]],
    },
    {
        "name": "self-correction",
        "turns": [
            "Assume the sun rises in the west. Where does the sun rise?",
            "Actually it rises in the east. Where does it rise?",
        ],
        "expected_keywords_per_turn": [[], ["east"]],
    },
]


def _has_keywords(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    low = text.lower()
    return all(k.lower() in low for k in keywords)


def default_contradiction_judge(a: str, b: str) -> bool:
    """Crude heuristic: flag contradiction if both have numbers that disagree."""
    from .metrics import extract_final_number

    an, bn = extract_final_number(a), extract_final_number(b)
    if an is None or bn is None:
        return False
    return an != bn


def evaluate(
    runner: Any,
    n_scenarios: int | None = None,
    num_diffusion_steps: int | None = None,
    judge: Callable[[str, str], bool] = default_contradiction_judge,
) -> dict[str, float]:
    scenarios = SCENARIOS if n_scenarios is None else SCENARIOS[:n_scenarios]
    all_passes: list[float] = []
    all_contras: list[float] = []
    for scen_idx, scen in enumerate(scenarios):
        sid = f"mt-{scen_idx}"
        responses: list[str] = []
        for turn in scen["turns"]:
            r = runner.chat(sid, turn, num_diffusion_steps=num_diffusion_steps)
            responses.append(r)
        # Keyword pass rate (1 if all turn-keywords matched).
        ok_turns = sum(
            1 for r, kw in zip(responses, scen["expected_keywords_per_turn"], strict=True)
            if _has_keywords(r, kw)
        )
        all_passes.append(ok_turns / max(1, len(responses)))
        # Contradiction rate among consecutive responses.
        if len(responses) > 1:
            cs = [
                1.0 if judge(a, b) else 0.0
                for a, b in zip(responses[:-1], responses[1:], strict=True)
            ]
            all_contras.append(sum(cs) / len(cs))
    return {
        "consistency": sum(all_passes) / max(1, len(all_passes)),
        "contradiction_rate": sum(all_contras) / max(1, len(all_contras)),
        "n": float(len(scenarios)),
    }
