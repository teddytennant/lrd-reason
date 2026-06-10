"""Packed raw-text dataset for the unsupervised pretrain (curriculum Stage 1).

Reads plain .txt files (the corpus mix documented in README "Training
curriculum": Nietzsche, Kant, mathlib4, arXiv math, systems source, FineWeb-Edu
slice), tokenizes each file with the provided encode function, concatenates the
id streams, and chops them into fixed-length blocks for next-token prediction.

Map-style and in-memory: the smoke corpus is tiny and even the full philosophy
corpus is a few hundred MB of ids. A streaming reader can replace this behind
the same interface if the cluster corpus outgrows RAM.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset


def byte_encode(text: str, vocab_size: int) -> list[int]:
    """Dependency-free byte-level tokenizer for the stub LLM path."""
    return [b % vocab_size for b in text.encode("utf-8")]


def resolve_corpus_paths(patterns: list[str], base_dir: str | Path = ".") -> list[Path]:
    """Expand a list of file paths / glob patterns relative to `base_dir`."""
    base = Path(base_dir)
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_absolute():
            matches = sorted(Path(m) for m in glob(pat))
        else:
            matches = sorted(Path(m) for m in glob(str(base / pat)))
        if not matches and p.is_file():
            matches = [p]
        out.extend(matches)
    files = [p for p in out if p.is_file()]
    if not files:
        raise FileNotFoundError(f"pretrain corpus is empty: no files match {patterns!r}")
    return files


class PackedTextDataset(Dataset):
    """Fixed-length blocks of token ids from a raw-text corpus.

    Each item is a LongTensor of shape [block_size]; the training loop uses it
    as both inputs and (shifted-internally) next-token labels.
    """

    def __init__(
        self,
        paths: list[Path],
        block_size: int,
        encode_fn: Callable[[str], list[int]],
    ) -> None:
        super().__init__()
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size}")
        ids: list[int] = []
        for p in paths:
            ids.extend(encode_fn(Path(p).read_text(encoding="utf-8", errors="replace")))
        n_blocks = len(ids) // block_size
        if n_blocks < 1:
            raise ValueError(
                f"corpus has {len(ids)} tokens, fewer than one block of {block_size}"
            )
        self._blocks = torch.tensor(
            ids[: n_blocks * block_size], dtype=torch.long
        ).view(n_blocks, block_size)

    def __len__(self) -> int:
        return self._blocks.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._blocks[idx]
