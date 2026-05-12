"""Latent-pair dataset (streaming JSONL).

Each line of the JSONL file is a JSON object:

    {
      "prompt":      str,
      "gold_cot":    str,
      "gold_answer": str,
      "session_id":  str,
      "turn_idx":    int
    }

Target latents are produced once by `scripts/encode_targets.py` and saved as a
side .pt file keyed by SHA-1 of the prompt. The dataset loads them lazily, falling
back to random latents when `latents_path` is None (smoke tests).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import IterableDataset


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


class LatentPairDataset(IterableDataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        latent_dim: int,
        latents_path: str | Path | None = None,
        task_embed_dim: int = 0,
    ) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.latent_dim = latent_dim
        self.task_embed_dim = task_embed_dim
        self._latents: dict[str, torch.Tensor] | None = None
        if latents_path is not None and Path(latents_path).exists():
            blob = torch.load(latents_path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict):
                self._latents = blob
            else:
                raise ValueError(f"latents file {latents_path} is not a dict")

    def _target_latent(self, prompt: str) -> torch.Tensor:
        if self._latents is not None:
            key = _prompt_hash(prompt)
            if key in self._latents:
                return self._latents[key].to(torch.float32)
        # Deterministic fallback for smoke tests: hash-seeded random.
        seed = int(_prompt_hash(prompt)[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.latent_dim, generator=g)

    def _task_embed(self, session_id: str) -> torch.Tensor:
        if self.task_embed_dim == 0:
            return torch.zeros(0)
        seed = int(_prompt_hash(session_id)[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(self.task_embed_dim, generator=g)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        with self.jsonl_path.open() as f:
            for i, line in enumerate(f):
                if worker_info is not None and i % worker_info.num_workers != worker_info.id:
                    continue
                row = json.loads(line)
                yield self._format(row)

    def _format(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": row["prompt"],
            "gold_cot": row.get("gold_cot", ""),
            "gold_answer": row.get("gold_answer", ""),
            "session_id": row.get("session_id", "0"),
            "turn_idx": int(row.get("turn_idx", 0)),
            "target_latent": self._target_latent(row["prompt"]),
            "task_embed": self._task_embed(row.get("session_id", "0")),
        }
