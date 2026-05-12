"""Curriculum Stage 1: unsupervised continued pretraining via the LoRA adapter.

Runs BEFORE the cold-start stages (`stage1.py` + `stage2.py`). The goal is to
imprint reasoning patterns and voice from real human truth-seekers rather than
from a synthetic teacher whose hedging would otherwise be baked into the model.

Trainable: the LoRA adapter on the frozen LLM. Recurrent state and diffusion
denoiser are not touched here; they enter training in the cold-start stages.

Data: raw text packed into fixed-length sequences. The intended corpus mix is
the seven sources documented in README.md ("Training curriculum"):
- Complete Nietzsche (Gutenberg Levy edition, all 17 books)
- Kant critical philosophy (Gutenberg: 3 Critiques + Grundlegung + Prolegomena + Perpetual Peace)
- Lean mathlib4 source
- arXiv math (filtered)
- SQLite, TigerBeetle, CPython source with comments
- FineWeb-Edu high-signal slice

Loss: standard next-token cross-entropy on packed sequences. No targets from a
generator; no CoT format expected at this stage.

To build on cluster. The implementation should mirror `stage1.py`'s structure
(build dataloader, build optimizer, train loop with atomic checkpointing via
`loop.py`) but load a raw-text dataset and freeze everything except the LoRA
params.
"""

from __future__ import annotations

import argparse

from ..config import RunSpec


def train(spec: RunSpec, resume: bool, max_steps: int | None = None) -> dict[str, float]:
    raise NotImplementedError(
        "Curriculum Stage 1 (unsupervised LoRA pretrain) is not yet implemented. "
        "See module docstring for the build plan."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    from ..config import load_config

    spec = load_config(args.config)
    train(spec, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
