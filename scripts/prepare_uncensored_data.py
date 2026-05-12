"""Build the SFT corpus for the 4B uncensored fine-tune track.

Mirrors lrd-reason's data philosophy (raw human truth-seeker text for voice,
verifier-grounded CoT for reasoning) but flattens it into a single chat-formatted
JSONL the standard SFTTrainer can consume — there is no recurrent/diffusion
pipeline on this track, just LoRA SFT on
HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive.

Inputs:
  --philosophy DIR    Gutenberg .txt files (Kant + Nietzsche). Stripped of the
                      Gutenberg license preamble and packed into ``chunk_tokens``
                      windows of raw text. Emitted as completion-only rows.
  --cot JSONL ...     One or more JSONLs in the format produced by
                      ``scripts/generate_data.py`` (keys: prompt, gold_cot,
                      gold_answer, ...). Emitted as chat rows under
                      ``COT_SYSTEM_PROMPT``.
  --output JSONL      Merged + shuffled corpus.
  --mix RATIO         Approximate fraction of *output rows* drawn from raw text
                      (default 0.35). The rest are CoT rows. CoT is the carrier
                      of reasoning competence; philosophy is the voice imprint.

Output rows are one of two shapes (the trainer dispatches on ``kind``):

  {"kind": "raw",  "text": "<packed chunk>"}
  {"kind": "chat", "messages": [{role, content}, ...]}

This is the single contract between data prep and ``finetune_uncensored.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

from lrd_reason.constitution import COT_SYSTEM_PROMPT

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S)
_WS = re.compile(r"[ \t]+")


def strip_gutenberg(text: str) -> str:
    """Drop Gutenberg license preamble/postamble. Idempotent on already-stripped text."""
    m_start = _GUTENBERG_START.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = _GUTENBERG_END.search(text)
    if m_end:
        text = text[: m_end.start()]
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs but keep paragraph breaks."""
    out_lines = []
    for line in text.splitlines():
        out_lines.append(_WS.sub(" ", line).rstrip())
    collapsed = "\n".join(out_lines)
    # Collapse 3+ blank lines to 2 (preserve paragraph structure).
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def chunk_words(text: str, chunk_words: int, stride_words: int) -> Iterator[str]:
    """Word-level chunking (cheap proxy for tokens; the tokenizer will trim later)."""
    if stride_words <= 0 or stride_words > chunk_words:
        raise ValueError("stride_words must be in (0, chunk_words]")
    words = text.split()
    if len(words) <= chunk_words:
        if words:
            yield " ".join(words)
        return
    i = 0
    while i < len(words):
        window = words[i : i + chunk_words]
        if len(window) < chunk_words // 4:
            break
        yield " ".join(window)
        i += stride_words


def iter_philosophy_rows(
    philosophy_dir: Path,
    chunk_words_n: int,
    stride_words: int,
) -> Iterator[dict]:
    files = sorted(p for p in philosophy_dir.iterdir() if p.suffix.lower() in {".txt", ".md"})
    if not files:
        raise FileNotFoundError(f"no .txt/.md files under {philosophy_dir}")
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="replace")
        cleaned = normalize_whitespace(strip_gutenberg(raw))
        for chunk in chunk_words(cleaned, chunk_words_n, stride_words):
            yield {"kind": "raw", "text": chunk, "source": p.name}


def cot_row_to_chat(row: dict) -> dict:
    """Turn a generate_data.py row into a chat-formatted SFT row.

    The assistant turn is gold_cot followed by the canonical 'Final answer: ...'
    suffix that ``COT_SYSTEM_PROMPT`` demands. If the generator already produced
    that suffix, we don't double it.
    """
    cot = row.get("gold_cot", "").rstrip()
    ans = row.get("gold_answer", "").strip()
    if ans and not re.search(r"(?im)^\s*(?:final\s*answer|answer)\s*:", cot):
        assistant = f"{cot}\nFinal answer: {ans}" if cot else f"Final answer: {ans}"
    else:
        assistant = cot if cot else ans
    return {
        "kind": "chat",
        "messages": [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": assistant},
        ],
        "source": row.get("source", "cot"),
    }


def iter_cot_rows(paths: list[Path]) -> Iterator[dict]:
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" not in row:
                    continue
                yield cot_row_to_chat(row)


def mix_and_write(
    raw_rows: list[dict],
    cot_rows: list[dict],
    out_path: Path,
    raw_fraction: float,
    seed: int,
) -> dict[str, int]:
    """Sample to hit raw_fraction on the larger pool; shuffle; write JSONL."""
    if not (0.0 <= raw_fraction <= 1.0):
        raise ValueError("raw_fraction must be in [0, 1]")
    rng = random.Random(seed)

    n_raw_have, n_cot_have = len(raw_rows), len(cot_rows)
    if n_raw_have == 0 and n_cot_have == 0:
        raise ValueError("both raw and cot pools are empty")

    # Solve for the largest mix that respects raw_fraction and available pools.
    # raw_fraction = n_raw / (n_raw + n_cot)  =>  n_cot = n_raw * (1 - r) / r
    if raw_fraction == 0.0 or n_raw_have == 0:
        n_raw, n_cot = 0, n_cot_have
    elif raw_fraction == 1.0 or n_cot_have == 0:
        n_raw, n_cot = n_raw_have, 0
    else:
        # Take all of the bottleneck side, derive the other.
        cap_from_raw = (n_raw_have, int(n_raw_have * (1 - raw_fraction) / raw_fraction))
        cap_from_cot = (int(n_cot_have * raw_fraction / (1 - raw_fraction)), n_cot_have)
        cand = [c for c in (cap_from_raw, cap_from_cot) if c[0] <= n_raw_have and c[1] <= n_cot_have]
        n_raw, n_cot = max(cand, key=lambda c: c[0] + c[1])

    picked = rng.sample(raw_rows, n_raw) + rng.sample(cot_rows, n_cot)
    rng.shuffle(picked)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"raw": n_raw, "cot": n_cot, "total": len(picked)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--philosophy", type=Path, default=None,
                    help="Dir of Gutenberg .txt files (Kant + Nietzsche). Omit to skip raw track.")
    ap.add_argument("--cot", type=Path, nargs="*", default=[],
                    help="JSONL files from scripts/generate_data.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mix", type=float, default=0.35,
                    help="Approximate fraction of OUTPUT rows from raw text (default 0.35)")
    ap.add_argument("--chunk-words", type=int, default=380,
                    help="Word window per raw chunk (~512 tokens for English prose)")
    ap.add_argument("--stride-words", type=int, default=380,
                    help="Stride between chunks; equal to chunk_words = no overlap")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    raw_rows: list[dict] = []
    if args.philosophy is not None:
        raw_rows = list(iter_philosophy_rows(args.philosophy, args.chunk_words, args.stride_words))

    cot_rows: list[dict] = list(iter_cot_rows(list(args.cot)))

    stats = mix_and_write(raw_rows, cot_rows, args.output, args.mix, args.seed)
    print(f"[prepare] raw={stats['raw']} cot={stats['cot']} total={stats['total']} -> {args.output}")


if __name__ == "__main__":
    main()
