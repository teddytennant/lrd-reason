"""CLI entry: `python -m lrd_reason.infer.cli --config configs/smoke.yaml --prompt ...`"""

from __future__ import annotations

import argparse
import uuid

from .pipeline import runner_from_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--stage1-ckpt", type=str, default=None)
    ap.add_argument("--stage2-ckpt", type=str, default=None)
    ap.add_argument("--sessions-dir", type=str, default="sessions")
    ap.add_argument("--session-id", type=str, default=None)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--diffusion-steps", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    runner = runner_from_config(
        args.config,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        sessions_dir=args.sessions_dir,
    )
    session_id = args.session_id or str(uuid.uuid4())
    print(f"session_id={session_id}")
    for i, p in enumerate(args.prompts):
        resp = runner.chat(
            session_id,
            p,
            num_diffusion_steps=args.diffusion_steps,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"--- turn {i} ---")
        print(f"PROMPT: {p}")
        print(f"REPLY:  {resp}")


if __name__ == "__main__":
    main()
