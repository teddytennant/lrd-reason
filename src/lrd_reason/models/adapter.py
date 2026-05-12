"""Soft-prefix adapter + LoRA attach helper.

The refined plan latent (256-d) is projected to `n_prefix * llm_hidden` and reshaped
to [B, n_prefix, llm_hidden]. This prefix is concatenated to the front of the LLM's
input embeddings. The LLM is otherwise frozen; only LoRA adapters on q/k/v/o/gate/
up/down_proj are trainable.

Why soft-prefix and not cross-attention: per Principle II, soft-prefix is one matmul
and a concat. Cross-attention into every layer is N times the surgery for marginal
PoC gain. If it later turns out a frozen prefix is insufficient, the upgrade path
is to add layer-wise prefix tuning, not full cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import AdapterConfig


class SoftPrefixAdapter(nn.Module):
    """Projects plan latent -> [B, n_prefix, llm_hidden] soft prefix."""

    def __init__(self, latent_dim: int, n_prefix: int, llm_hidden: int) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.llm_hidden = llm_hidden
        self.proj = nn.Linear(latent_dim, n_prefix * llm_hidden)
        # Initialize close to zero so an untrained adapter is a near-no-op (lets the
        # frozen LLM behave like itself when training starts).
        with torch.no_grad():
            self.proj.weight.mul_(0.01)
            self.proj.bias.zero_()

    def forward(self, plan_latent: torch.Tensor) -> torch.Tensor:
        B = plan_latent.shape[0]
        return self.proj(plan_latent).view(B, self.n_prefix, self.llm_hidden)


def attach_lora(llm: nn.Module, cfg: AdapterConfig) -> nn.Module:
    """Wrap an LLM with LoRA adapters on the configured target modules.

    Returns the wrapped model. Original weights are frozen by peft. If lora_r == 0
    or peft is unavailable, returns the LLM unchanged.
    """
    if cfg.lora_r == 0:
        return llm
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA attachment. Install with `pip install '.[hf]'`."
        ) from e
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(llm, lora_cfg)


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
