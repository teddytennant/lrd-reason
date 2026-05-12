"""Per-domain verifiers for the RLVR main phase.

Each verifier is a pure function `(predicted_answer, gold_answer) -> float in [0, 1]`
that scores whether the model's final answer matches the ground truth for its
domain. The verifiers are the reward signal for `train/stage_rlvr.py` — there is
no learned reward model anywhere in this project, by design (see README "Project
Values" and the discussion of value drift in synthetic data).

Planned verifiers:
- math: numeric / sympy-equivalence on a parsed answer expression
- code: pytest pass rate against a held-out test suite
- logic: exact-match on a canonicalised answer
- multi_hop_qa: exact-match on the gold span (with light normalisation)
- formal: Lean / Coq type-check on a submitted proof term

To build on cluster.
"""

from __future__ import annotations

__all__: list[str] = []
