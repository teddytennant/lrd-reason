"""Stage-2 cold-start CoT trace generator.

Library code. Not auto-run on import. `scripts/generate_data.py` is the H200 entry
point. Produces the small (~10–50k example) cold-start corpus the recurrent +
diffusion modules need before RLVR can train stably; bulk reasoning competence
comes from Stage 3, not from here.

The generator calls a vLLM engine with `COT_SYSTEM_PROMPT` as the system prompt
and writes one JSONL row per problem. The vLLM engine is injected (no auto-load)
so this file is importable on CPU/laptops without `pip install vllm`. Smoke tests
use a fake engine.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..constitution import COT_SYSTEM_PROMPT, cot_messages


@runtime_checkable
class ChatEngine(Protocol):
    """Anything with a `.chat(messages_list) -> [{'text': str}, ...]` method.

    vLLM's `LLM.chat()` matches this. So does a test fake.
    """

    def chat(self, messages_list: list[list[dict]], **kwargs: Any) -> list[Any]:
        ...


_ANSWER_RE = re.compile(r"(?im)^\s*(?:final\s*answer|answer)\s*:\s*(.+?)\s*$")


def split_cot_answer(text: str) -> tuple[str, str]:
    """Heuristic: split the assistant output into (cot, answer).

    Looks for a trailing 'Final answer:' or 'Answer:' marker; falls back to the
    last non-empty line as the answer.
    """
    m = list(_ANSWER_RE.finditer(text))
    if m:
        last = m[-1]
        return text[: last.start()].rstrip(), last.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return text.strip(), ""
    return "\n".join(lines[:-1]), lines[-1]


class CoTGenerator:
    def __init__(
        self,
        engine: ChatEngine,
        system_prompt: str = COT_SYSTEM_PROMPT,
        message_builder: Callable[[str], list[dict]] = cot_messages,
    ) -> None:
        self.engine = engine
        self.system_prompt = system_prompt
        self.message_builder = message_builder

    def _build(self, problem: str) -> list[dict]:
        msgs = self.message_builder(problem)
        # Ensure the configured system prompt is in front (override builder default).
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = self.system_prompt
        else:
            msgs = [{"role": "system", "content": self.system_prompt}, *msgs]
        return msgs

    def generate(
        self,
        problems: list[str],
        session_ids: list[str] | None = None,
        turn_idxs: list[int] | None = None,
        **engine_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if session_ids is None:
            session_ids = [str(i) for i in range(len(problems))]
        if turn_idxs is None:
            turn_idxs = [0] * len(problems)
        messages_list = [self._build(p) for p in problems]
        outputs = self.engine.chat(messages_list, **engine_kwargs)
        rows: list[dict[str, Any]] = []
        for problem, sess, turn, out in zip(problems, session_ids, turn_idxs, outputs, strict=True):
            text = _extract_text(out)
            cot, answer = split_cot_answer(text)
            rows.append(
                {
                    "prompt": problem,
                    "gold_cot": cot,
                    "gold_answer": answer,
                    "session_id": sess,
                    "turn_idx": turn,
                }
            )
        return rows

    def generate_to_jsonl(
        self,
        problems: list[str],
        out_path: str | Path,
        session_ids: list[str] | None = None,
        turn_idxs: list[int] | None = None,
        append: bool = True,
        batch_size: int = 64,
        **engine_kwargs: Any,
    ) -> int:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        written = 0
        with out_path.open(mode) as f:
            for i in range(0, len(problems), batch_size):
                batch_problems = problems[i : i + batch_size]
                batch_sess = (
                    session_ids[i : i + batch_size] if session_ids is not None else None
                )
                batch_turns = (
                    turn_idxs[i : i + batch_size] if turn_idxs is not None else None
                )
                rows = self.generate(
                    batch_problems, batch_sess, batch_turns, **engine_kwargs
                )
                for row in rows:
                    f.write(json.dumps(row) + "\n")
                    written += 1
                f.flush()
        return written


def _extract_text(out: Any) -> str:
    """Best-effort text extraction across vLLM/dict/string outputs."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        if "text" in out:
            return out["text"]
        if "outputs" in out and out["outputs"]:
            first = out["outputs"][0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
    # vLLM RequestOutput shape
    if hasattr(out, "outputs") and out.outputs:
        return out.outputs[0].text
    return str(out)
