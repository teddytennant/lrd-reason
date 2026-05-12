"""Pure-Python tests for the 4B uncensored fine-tune track.

We exercise the data prep helpers and the row-formatting hook directly; the
training script itself is NOT invoked (it requires CUDA + the model download
and isn't suitable for the CPU pytest contract).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Load scripts/<name>.py as a module.

    Register in sys.modules before exec_module so dataclass decorators can resolve
    cls.__module__ (sys.modules.get(name).__dict__).
    """
    import sys

    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prep():
    return _load_script("prepare_uncensored_data")


@pytest.fixture(scope="module")
def finetune():
    return _load_script("finetune_uncensored")


class _FakeTokenizer:
    """Minimal duck-typed stand-in for an HF tokenizer."""

    eos_token = "<|endoftext|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        return "".join(parts)


# ---------- prepare_uncensored_data ----------

def test_strip_gutenberg_idempotent(prep):
    body = "Beyond Good and Evil. The text proper. End."
    framed = (
        "Header trash\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK BEYOND GOOD AND EVIL ***\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK BEYOND GOOD AND EVIL ***\n"
        "License footer"
    )
    once = prep.strip_gutenberg(framed)
    twice = prep.strip_gutenberg(once)
    assert "Header trash" not in once
    assert "License footer" not in once
    assert body in once
    assert once == twice  # idempotent


def test_normalize_whitespace(prep):
    inp = "foo    bar\t\tbaz\n\n\n\nnext para"
    out = prep.normalize_whitespace(inp)
    assert "foo bar baz" in out
    assert "\n\n\n" not in out


def test_chunk_words_respects_stride(prep):
    text = " ".join(str(i) for i in range(80))
    chunks = list(prep.chunk_words(text, chunk_words=40, stride_words=40))
    assert len(chunks) == 2
    assert chunks[0].startswith("0 1 2")
    assert chunks[1].startswith("40 41")


def test_chunk_words_keeps_long_trailing_partial(prep):
    # 100 words at chunk=40/stride=40: 0-39, 40-79, 80-99. The 20-word tail is
    # >= chunk_words/4 (=10), so it survives.
    text = " ".join(str(i) for i in range(100))
    chunks = list(prep.chunk_words(text, chunk_words=40, stride_words=40))
    assert len(chunks) == 3
    assert chunks[-1].startswith("80")


def test_chunk_words_short_text_yields_one(prep):
    chunks = list(prep.chunk_words("a b c", chunk_words=10, stride_words=10))
    assert chunks == ["a b c"]


def test_cot_row_to_chat_adds_final_answer(prep):
    row = {"prompt": "What is 6*7?", "gold_cot": "Multiply.", "gold_answer": "42"}
    out = prep.cot_row_to_chat(row)
    assert out["kind"] == "chat"
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["content"].endswith("Final answer: 42")


def test_cot_row_to_chat_preserves_existing_marker(prep):
    row = {
        "prompt": "x",
        "gold_cot": "Reason.\nFinal answer: 1",
        "gold_answer": "1",
    }
    out = prep.cot_row_to_chat(row)
    # The renderer must not double the marker.
    assistant = out["messages"][-1]["content"]
    assert assistant.count("Final answer:") == 1


def test_iter_philosophy_rows(prep, tmp_path):
    (tmp_path / "nietzsche.txt").write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        + " ".join(["word"] * 200)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
    )
    rows = list(prep.iter_philosophy_rows(tmp_path, chunk_words_n=50, stride_words=50))
    assert len(rows) >= 3
    assert all(r["kind"] == "raw" for r in rows)
    assert all(r["source"] == "nietzsche.txt" for r in rows)


def test_mix_and_write_respects_fraction(prep, tmp_path):
    raw = [{"kind": "raw", "text": f"r{i}"} for i in range(100)]
    cot = [{"kind": "chat", "messages": [{"role": "user", "content": f"c{i}"}]} for i in range(100)]
    out = tmp_path / "mix.jsonl"
    stats = prep.mix_and_write(raw, cot, out, raw_fraction=0.25, seed=0)
    assert stats["raw"] + stats["cot"] == stats["total"]
    # Allow ±1 row of rounding slack.
    assert abs(stats["raw"] / stats["total"] - 0.25) <= 0.02
    # Output is shuffled JSONL.
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == stats["total"]


def test_mix_and_write_all_cot_when_no_raw(prep, tmp_path):
    cot = [{"kind": "chat", "messages": [{"role": "user", "content": "x"}]}]
    out = tmp_path / "mix.jsonl"
    stats = prep.mix_and_write([], cot, out, raw_fraction=0.5, seed=0)
    assert stats["raw"] == 0 and stats["cot"] == 1


# ---------- finetune_uncensored ----------

def test_format_row_chat(finetune):
    row = {
        "kind": "chat",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
    }
    text = finetune.format_row(row, _FakeTokenizer())
    assert "<system>S</system>" in text
    assert "<assistant>A</assistant>" in text


def test_format_row_raw_appends_eos(finetune):
    text = finetune.format_row({"kind": "raw", "text": "hello world"}, _FakeTokenizer())
    assert text.startswith("hello world")
    assert text.rstrip().endswith("<|endoftext|>")


def test_format_row_unknown_kind_raises(finetune):
    with pytest.raises(ValueError):
        finetune.format_row({"kind": "bogus"}, _FakeTokenizer())


def test_spec_from_yaml_roundtrip(finetune, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model_id: foo/bar\n"
        "lora_r: 8\n"
        "num_train_epochs: 0.5\n"
    )
    spec = finetune.FineTuneSpec.from_yaml(cfg)
    assert spec.model_id == "foo/bar"
    assert spec.lora_r == 8
    assert spec.num_train_epochs == 0.5
    # Defaults preserved.
    assert spec.lora_targets and "q_proj" in spec.lora_targets


def test_load_jsonl_skips_blank(finetune, tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n')
    rows = finetune.load_jsonl(p)
    assert rows == [{"a": 1}, {"a": 2}]
