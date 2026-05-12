"""Rectified-flow latent denoiser + sampler + loss.

Rectified flow (Liu et al. 2022) trains a velocity field v(x_t, t, cond) such that the
straight-line trajectory from noise (t=0) to data (t=1) has constant velocity x1 - x0.
At inference, integrate with Euler steps; K=4-8 typically suffices.

Operating in 256-dim latent space (no spatial structure), so the denoiser is a small
transformer over a single "token" with conditioning via FiLM.

Shapes:
    x_t       : [B, latent_dim]
    t         : [B] in (0, 1)
    cond_state: [B, hidden_dim] = recurrent state
    cond_z    : [B, latent_dim] = encoded prompt latent
    cond_task : [B, task_dim] or None
    -> velocity: [B, latent_dim]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import DiffusionConfig


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style timestep embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class FiLMBlock(nn.Module):
    """FiLM-modulated transformer block. Single-token sequence."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Single-token attention degenerates; we use it as a residual MLP-only block
        # with a wider hidden dim, plus FiLM modulation.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.film_scale = nn.Linear(d_model, d_model)
        self.film_shift = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, d_model], cond: [B, d_model]
        scale = self.film_scale(cond)
        shift = self.film_shift(cond)
        h = self.norm1(x) * (1 + scale) + shift
        h = self.mlp(self.norm2(h))
        return x + h


class RectifiedFlowDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_state_dim: int,
        cond_task_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.x_proj = nn.Linear(latent_dim, d_model)
        self.t_proj = nn.Linear(d_model, d_model)
        self.state_proj = nn.Linear(cond_state_dim, d_model)
        self.z_proj = nn.Linear(latent_dim, d_model)
        self.task_proj = nn.Linear(cond_task_dim, d_model) if cond_task_dim > 0 else None

        self.blocks = nn.ModuleList(
            [FiLMBlock(d_model=d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_state: torch.Tensor,
        cond_z: torch.Tensor,
        cond_task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.x_proj(x_t)
        t_emb = _sinusoidal_timestep_embedding(t, self.d_model)
        cond = self.t_proj(t_emb) + self.state_proj(cond_state) + self.z_proj(cond_z)
        if self.task_proj is not None and cond_task is not None:
            cond = cond + self.task_proj(cond_task)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(self.out_norm(h))


def rectified_flow_loss(
    model: RectifiedFlowDenoiser,
    x0: torch.Tensor,
    cond_state: torch.Tensor,
    cond_z: torch.Tensor,
    cond_task: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard rectified-flow regression loss.

    Sample t ~ U(0, 1), eps ~ N(0, I). Construct x_t = (1 - t) * eps + t * x0.
    Target velocity v* = x0 - eps. Predict v_hat = model(x_t, t, cond). MSE.
    """
    B = x0.shape[0]
    t = torch.rand(B, device=x0.device)
    eps = torch.randn_like(x0)
    t_expand = t.unsqueeze(-1)
    x_t = (1 - t_expand) * eps + t_expand * x0
    v_target = x0 - eps
    v_pred = model(x_t, t, cond_state, cond_z, cond_task)
    return nn.functional.mse_loss(v_pred, v_target)


class RFSampler:
    """Euler integration of the rectified flow from noise (t=0) to data (t=1)."""

    def __init__(self, num_steps: int) -> None:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        model: RectifiedFlowDenoiser,
        init_noise: torch.Tensor,
        cond_state: torch.Tensor,
        cond_z: torch.Tensor,
        cond_task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = init_noise
        dt = 1.0 / self.num_steps
        for i in range(self.num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = model(x, t, cond_state, cond_z, cond_task)
            x = x + dt * v
        return x
