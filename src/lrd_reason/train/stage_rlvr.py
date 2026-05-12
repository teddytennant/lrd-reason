"""Curriculum Stage 3: RLVR (Reinforcement Learning from Verifiable Rewards).

Runs AFTER the cold-start stages (`stage1.py` + `stage2.py`). The cold-start
checkpoint provides both the initial policy and the KL reference; reward comes
from per-domain verifiers in `lrd_reason/eval/verifiers/`, not a learned reward
model — this is what makes the value imprint grounded in reality rather than
in a teacher's posture.

Trainable: all the same modules the cold-start trained (recurrent + diffusion +
adapter + LoRA), now with policy-gradient signal.

Algorithm: GRPO-style group-relative advantages without a separate value head.
For each problem in a batch, sample K rollouts at temperature > 0, score each
rollout via the appropriate verifier, normalise rewards within the group, then
take a clipped policy-gradient step.

Reward shape per rollout = verifier_pass
                         + format_bonus  (final "Final answer:" marker present)
                         - length_penalty (per-token over budget)
                         - hedge_penalty  (regex hits on "I'd suggest", "perhaps", etc.)
KL anchor: against the Stage-2 checkpoint, weight ~0.01–0.05 to start.

Verifiers (per problem domain, see `lrd_reason/eval/verifiers/`):
- math: numeric equivalence via sympy
- code: pytest pass rate against held-out tests
- logic: exact-match
- multi-hop QA: exact-match on gold span
- formal (mathlib4): Lean type-check

Caveats for this architecture:
- The recurrent state can develop pathological hidden trajectories under RL; the
  KL anchor against the cold-start is essential.
- The diffusion denoiser introduces variance into rollouts; expect to need lower
  learning rates and more rollouts per update than standard GRPO on a flat LLM.

To build on cluster.
"""

from __future__ import annotations

import argparse

from ..config import RunSpec


def train(
    spec: RunSpec,
    resume: bool,
    cold_start_checkpoint: str | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    raise NotImplementedError(
        "Curriculum Stage 3 (RLVR) is not yet implemented. "
        "See module docstring for the algorithm and reward shape."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--cold-start-checkpoint", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    from ..config import load_config

    spec = load_config(args.config)
    train(
        spec,
        resume=args.resume,
        cold_start_checkpoint=args.cold_start_checkpoint,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
