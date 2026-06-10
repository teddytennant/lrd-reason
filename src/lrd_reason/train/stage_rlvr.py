"""Curriculum Stage 3: RLVR (policy gradient from verifiable verifiers, no learned RM).

Stub. Runs after cold-start; uses stage2 ckpt as KL ref + verifiers/ for reward.
See README "Training curriculum" (Stage 3) and eval/verifiers/.

Raises NotImplementedError until the cluster build.
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
    raise NotImplementedError("Stage 3 RLVR is a stub (see README).")


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
