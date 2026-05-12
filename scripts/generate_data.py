"""H200 entry: generate CoT traces from the frozen LLM via vLLM.

Loads problems from a JSONL (one {"prompt": str, "session_id": ..., "turn_idx": ...}
per line), spins up a vLLM engine with the configured backbone, and writes the
output JSONL.

Not executed by tests; this is the script that bakes the data on the H200 box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lrd_reason.config import load_config
from lrd_reason.constitution import COT_SYSTEM_PROMPT
from lrd_reason.data.cot_generator import CoTGenerator


def _load_problems(path: Path, shard: int, num_shards: int, target: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i % num_shards != shard:
                continue
            rows.append(json.loads(line))
            if len(rows) >= target:
                break
    return rows


def _build_engine(hf_id: str, trust_remote_code: bool):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. `pip install '.[vllm]'`.", file=sys.stderr)
        sys.exit(1)

    class _Engine:
        def __init__(self):
            self.llm = LLM(model=hf_id, trust_remote_code=trust_remote_code, dtype="auto")
            self.sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)

        def chat(self, messages_list, **kwargs):
            return self.llm.chat(messages_list, sampling_params=self.sp)

    return _Engine()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--problems", required=True, type=str, help="JSONL with one problem per line")
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--resume", action="store_true", help="append to output if it exists")
    args = ap.parse_args()

    spec = load_config(args.config)
    problems = _load_problems(Path(args.problems), args.shard, args.num_shards, args.target)

    # Resume: count existing lines, skip them.
    out_path = Path(args.output)
    skip = 0
    if args.resume and out_path.exists():
        with out_path.open() as f:
            skip = sum(1 for _ in f)
        print(f"[resume] skipping {skip} already-generated rows")
    problems = problems[skip:]

    engine = _build_engine(spec.model.llm.hf_id, spec.model.llm.trust_remote_code)
    gen = CoTGenerator(engine=engine, system_prompt=COT_SYSTEM_PROMPT)
    n = gen.generate_to_jsonl(
        problems=[r["prompt"] for r in problems],
        out_path=out_path,
        session_ids=[r.get("session_id", str(i)) for i, r in enumerate(problems)],
        turn_idxs=[r.get("turn_idx", 0) for r in problems],
        append=args.resume,
    )
    print(f"[done] wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
