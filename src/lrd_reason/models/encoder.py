"""Frozen text encoders + projector down to the working latent dim.

Three implementations behind a common Protocol:

- StubEncoder   — deterministic hash-based, no deps. Used in CPU smoke tests.
- BGEEncoder    — sentence-transformers BAAI/bge-large-en-v1.5. Default for real runs.
- LingbotEncoder— transformers AutoModel for robbyant/lingbot-world-base-cam. Warns on
                  construction: that checkpoint expects images, will not produce
                  meaningful text embeddings.

EncoderProjector projects raw_dim -> latent_dim with a frozen linear. Init can be
random orthogonal (default) or loaded from a PCA-fitted .pt file produced by
scripts/fit_projector.py (not in this build; see LAUNCH.md).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from ..config import EncoderConfig


@runtime_checkable
class Encoder(Protocol):
    raw_dim: int

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Returns Tensor of shape [B, raw_dim]."""
        ...


class StubEncoder:
    """Deterministic hash-based encoder. No network, no weights. CPU-only.

    Maps each text to a fixed vector via SHA-256 -> bytes -> float -> reshape.
    Stable across processes and platforms.
    """

    def __init__(self, raw_dim: int) -> None:
        self.raw_dim = raw_dim

    def _vec(self, text: str) -> torch.Tensor:
        # SHA-256 gives 32 bytes. Tile by re-hashing until we have raw_dim bytes,
        # then map each byte to a float in [-1, 1] (avoids NaN traps that come from
        # reinterpreting raw bytes as float32).
        chunks = []
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        while len(b"".join(chunks)) < self.raw_dim:
            chunks.append(seed)
            seed = hashlib.sha256(seed).digest()
        buf = b"".join(chunks)[: self.raw_dim]
        arr = torch.tensor([b / 127.5 - 1.0 for b in buf], dtype=torch.float32)
        arr = arr - arr.mean()
        n = arr.norm().clamp(min=1e-6)
        return arr / n

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.stack([self._vec(t) for t in texts], dim=0)


class BGEEncoder:
    """Frozen BAAI/bge-large-en-v1.5 wrapper. Lazy-loads sentence-transformers."""

    def __init__(self, hf_id: str, raw_dim: int, pooling: str = "cls", device: str = "cpu") -> None:
        self.hf_id = hf_id
        self.raw_dim = raw_dim
        self.pooling = pooling
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for BGEEncoder. "
                "Install with `pip install '.[hf]'`."
            ) from e
        self._model = SentenceTransformer(self.hf_id, device=self.device)
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        self._ensure_loaded()
        out = self._model.encode(  # type: ignore[union-attr]
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if out.shape[-1] != self.raw_dim:
            raise ValueError(
                f"BGE returned dim {out.shape[-1]}, config says raw_dim={self.raw_dim}"
            )
        return out


class LingbotEncoder:
    """Frozen robbyant/lingbot-world-base-cam wrapper.

    Honest disclosure: this checkpoint is a camera-based visual world model. Feeding
    text into it will not produce semantically meaningful embeddings for reasoning
    tasks. This wrapper exists so a future visual-reasoning extension can swap in
    here without touching pipeline code. Default for math/CoT data is BGEEncoder.
    """

    def __init__(self, hf_id: str, raw_dim: int, device: str = "cpu") -> None:
        warnings.warn(
            "LingbotEncoder: lingbot-world-base-cam is a camera-based visual world model. "
            "Text embeddings produced by this wrapper are not semantically meaningful. "
            "Use BGEEncoder for math/CoT text data.",
            UserWarning,
            stacklevel=2,
        )
        self.hf_id = hf_id
        self.raw_dim = raw_dim
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for LingbotEncoder. Install with `pip install '.[hf]'`."
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.hf_id, trust_remote_code=True).to(self.device)
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        self._ensure_loaded()
        toks = self._tokenizer(  # type: ignore[union-attr]
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self._model(**toks)  # type: ignore[misc]
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return h.mean(dim=1)


def build_encoder(cfg: EncoderConfig, device: str = "cpu") -> Encoder:
    if cfg.kind == "stub":
        return StubEncoder(raw_dim=cfg.raw_dim)
    if cfg.kind == "bge":
        if cfg.hf_id is None:
            raise ValueError("encoder.hf_id required for kind=bge")
        return BGEEncoder(hf_id=cfg.hf_id, raw_dim=cfg.raw_dim, pooling=cfg.pooling, device=device)
    if cfg.kind == "lingbot":
        if cfg.hf_id is None:
            raise ValueError("encoder.hf_id required for kind=lingbot")
        return LingbotEncoder(hf_id=cfg.hf_id, raw_dim=cfg.raw_dim, device=device)
    raise ValueError(f"unknown encoder kind: {cfg.kind}")


class EncoderProjector(nn.Module):
    """Frozen linear projection raw_dim -> latent_dim.

    Init options:
      - None (default): random orthogonal init via torch.nn.init.orthogonal_.
      - path: load a torch.save'd tensor of shape [latent_dim, raw_dim] from disk.

    Marked non-trainable on construction.
    """

    def __init__(self, raw_dim: int, latent_dim: int, projector_path: str | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(raw_dim, latent_dim, bias=False)
        if projector_path is not None and Path(projector_path).exists():
            w = torch.load(projector_path, map_location="cpu", weights_only=False)
            if w.shape != (latent_dim, raw_dim):
                raise ValueError(
                    f"projector weight shape {tuple(w.shape)} != expected "
                    f"({latent_dim}, {raw_dim})"
                )
            with torch.no_grad():
                self.proj.weight.copy_(w)
        else:
            with torch.no_grad():
                nn.init.orthogonal_(self.proj.weight)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
