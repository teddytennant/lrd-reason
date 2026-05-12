"""Persistent recurrent state module.

Two implementations behind a common nn.Module interface:

- GRUBlock   — pure torch, CPU-runnable, default for smoke tests.
- Mamba2Block — requires `mamba-ssm`. Used in real H200 runs.

Both consume a sequence of latent vectors plus a prior state, and emit a new state.
Persistence is handled at a higher level (infer/state_store.py); this module just
exposes the state tensor.

Input/output dims:
    z         : [B, latent_dim]            current observation
    prev_state: [B, hidden_dim]            persistent state (or None)
    task_embed: [B, task_embed_dim] or None
    -> new_state: [B, hidden_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RecurrentConfig


class GRUBlock(nn.Module):
    """Simple GRUCell stack. CPU-friendly. hidden_dim must equal latent_dim."""

    def __init__(self, latent_dim: int, hidden_dim: int, n_layers: int, task_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim != latent_dim:
            raise ValueError(f"GRUBlock requires hidden_dim ({hidden_dim}) == latent_dim ({latent_dim})")
        self.hidden_dim = hidden_dim
        self.task_proj = nn.Linear(task_dim, latent_dim) if task_dim > 0 else None
        self.cells = nn.ModuleList(
            [nn.GRUCell(input_size=latent_dim, hidden_size=hidden_dim) for _ in range(n_layers)]
        )

    def forward(
        self,
        z: torch.Tensor,
        prev_state: torch.Tensor | None,
        task_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = z.shape[0]
        if prev_state is None:
            prev_state = z.new_zeros(B, self.hidden_dim)
        h = z
        if self.task_proj is not None and task_embed is not None:
            h = h + self.task_proj(task_embed)
        state = prev_state
        for cell in self.cells:
            state = cell(h, state)
            h = state
        return state

    def init_state(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


class Mamba2Block(nn.Module):
    """Mamba-2 state-space block. Requires `mamba-ssm`.

    Falls back to ImportError on construction if mamba-ssm is unavailable.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        expand: int,
        n_layers: int,
        task_dim: int = 0,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as e:
            raise ImportError(
                "mamba-ssm is required for Mamba2Block. "
                "Install with `pip install '.[mamba]'` (CUDA-only)."
            ) from e

        if hidden_dim != latent_dim:
            raise ValueError("Mamba2Block requires hidden_dim == latent_dim")
        self.hidden_dim = hidden_dim
        self.task_proj = nn.Linear(task_dim, latent_dim) if task_dim > 0 else None
        self.blocks = nn.ModuleList(
            [Mamba2(d_model=latent_dim, expand=expand) for _ in range(n_layers)]
        )
        self.gate = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self,
        z: torch.Tensor,
        prev_state: torch.Tensor | None,
        task_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Mamba2 operates on sequences. We synthesize a length-2 sequence
        # [prev_state, z] and read out the final position.
        B = z.shape[0]
        if prev_state is None:
            prev_state = z.new_zeros(B, self.hidden_dim)
        h = z
        if self.task_proj is not None and task_embed is not None:
            h = h + self.task_proj(task_embed)
        seq = torch.stack([prev_state, h], dim=1)  # [B, 2, D]
        for block in self.blocks:
            seq = seq + block(seq)
        new_state = seq[:, -1]
        gated = self.gate(torch.cat([prev_state, new_state], dim=-1))
        return torch.sigmoid(gated) * new_state + (1 - torch.sigmoid(gated)) * prev_state

    def init_state(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)


def build_recurrent(cfg: RecurrentConfig, latent_dim: int, task_dim: int = 0) -> nn.Module:
    if cfg.kind == "gru":
        return GRUBlock(
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            task_dim=task_dim,
        )
    if cfg.kind == "mamba2":
        return Mamba2Block(
            latent_dim=latent_dim,
            hidden_dim=cfg.hidden_dim,
            expand=cfg.expand,
            n_layers=cfg.n_layers,
            task_dim=task_dim,
        )
    raise ValueError(f"unknown recurrent kind: {cfg.kind}")


def contrastive_state_loss(
    states_t: torch.Tensor, states_tp1: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE on consecutive states within a batch.

    states_t[i] should be closer to states_tp1[i] than to states_tp1[j != i].
    Anti-collapse signal: penalizes states that ignore the current observation.

    Both inputs: [B, D]. Returns a scalar.
    """
    z1 = nn.functional.normalize(states_t, dim=-1)
    z2 = nn.functional.normalize(states_tp1, dim=-1)
    logits = z1 @ z2.t() / temperature
    targets = torch.arange(z1.shape[0], device=z1.device)
    return nn.functional.cross_entropy(logits, targets)
