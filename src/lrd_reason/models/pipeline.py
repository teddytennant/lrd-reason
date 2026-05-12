"""LRDPipeline — wires encoder, recurrent state, diffusion, adapter, and LLM together.

Forward modes:
  - training: returns dict of losses (diffusion + contrastive + optional SFT NLL).
  - inference: encode -> recurrent step -> RF refinement -> adapter -> LLM generate.

Ablation flags (from AblationConfig) gate components:
  - !use_recurrent: state is zeros, not updated.
  - !use_diffusion: plan_latent = encoded prompt latent (no refinement).
  - zero_prefix:    soft prefix is zero (baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import AblationConfig, ModelConfig
from .adapter import SoftPrefixAdapter, attach_lora
from .diffusion import RectifiedFlowDenoiser, RFSampler, rectified_flow_loss
from .encoder import EncoderProjector, build_encoder
from .llm import build_llm
from .recurrent_state import build_recurrent, contrastive_state_loss


@dataclass
class PipelineOutputs:
    plan_latent: torch.Tensor
    state: torch.Tensor
    response_ids: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] | None = None


class LRDPipeline(nn.Module):
    def __init__(
        self,
        model_cfg: ModelConfig,
        ablation: AblationConfig,
        task_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.ablation = ablation
        self.latent_dim = model_cfg.latent_dim
        self.task_dim = task_dim

        # Encoder (frozen, not an nn.Module submodule — kept outside state_dict).
        self.encoder = build_encoder(model_cfg.encoder, device=device)
        self.projector = EncoderProjector(
            raw_dim=model_cfg.encoder.raw_dim,
            latent_dim=model_cfg.latent_dim,
            projector_path=model_cfg.encoder.projector_path,
        )

        # Recurrent state (trainable).
        self.recurrent = build_recurrent(
            model_cfg.recurrent, latent_dim=model_cfg.latent_dim, task_dim=task_dim
        )

        # Diffusion denoiser (trainable).
        self.denoiser = RectifiedFlowDenoiser(
            latent_dim=model_cfg.latent_dim,
            cond_state_dim=model_cfg.recurrent.hidden_dim,
            cond_task_dim=task_dim,
            d_model=model_cfg.diffusion.d_model,
            n_layers=model_cfg.diffusion.n_layers,
            n_heads=model_cfg.diffusion.n_heads,
            dropout=model_cfg.diffusion.dropout,
        )
        self.sampler = RFSampler(num_steps=model_cfg.diffusion.n_inference_steps)

        # LLM (frozen; LoRA attached below). Hidden size needs to match adapter.
        if model_cfg.llm.kind == "stub":
            llm_hidden = model_cfg.adapter.llm_hidden
            self.llm = build_llm(model_cfg.llm, hidden_size=llm_hidden, device=device, dtype=dtype)
        else:
            self.llm = build_llm(model_cfg.llm, hidden_size=0, device=device, dtype=dtype)
            llm_hidden = self.llm.hidden_size  # type: ignore[attr-defined]
        self.llm = attach_lora(self.llm, model_cfg.adapter)

        # Adapter (trainable).
        self.adapter = SoftPrefixAdapter(
            latent_dim=model_cfg.latent_dim,
            n_prefix=model_cfg.adapter.n_prefix,
            llm_hidden=llm_hidden,
        )

    # ---- core sub-steps ----

    def encode(self, prompts: list[str], device: torch.device | str = "cpu") -> torch.Tensor:
        raw = self.encoder.encode(prompts).to(device)
        return self.projector(raw)

    def step_recurrent(
        self,
        z: torch.Tensor,
        prev_state: torch.Tensor | None,
        task_embed: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.ablation.use_recurrent:
            B = z.shape[0]
            return z.new_zeros(B, self.model_cfg.recurrent.hidden_dim)
        return self.recurrent(z, prev_state, task_embed)

    def refine(
        self,
        z: torch.Tensor,
        state: torch.Tensor,
        task_embed: torch.Tensor | None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if not self.ablation.use_diffusion:
            return z
        steps = num_steps or self.sampler.num_steps
        sampler = RFSampler(num_steps=steps)
        init_noise = torch.randn_like(z)
        return sampler.sample(self.denoiser, init_noise, state, z, task_embed)

    def make_prefix(self, plan_latent: torch.Tensor) -> torch.Tensor:
        if self.ablation.zero_prefix:
            B = plan_latent.shape[0]
            return plan_latent.new_zeros(
                B, self.model_cfg.adapter.n_prefix, self.model_cfg.adapter.llm_hidden
            )
        return self.adapter(plan_latent)

    # ---- end-to-end inference ----

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        input_ids: torch.Tensor,
        prev_state: torch.Tensor | None = None,
        task_embed: torch.Tensor | None = None,
        num_diffusion_steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = input_ids.device
        z = self.encode(prompts, device=device)
        state = self.step_recurrent(z, prev_state, task_embed)
        plan = self.refine(z, state, task_embed, num_steps=num_diffusion_steps)
        prefix = self.make_prefix(plan)
        prefix = prefix.to(device=device, dtype=self.llm.embed(input_ids[:1, :1]).dtype)
        out_ids = self.llm.generate(prefix, input_ids, max_new_tokens=max_new_tokens)
        return out_ids, state, plan

    # ---- training forward ----

    def training_step(
        self,
        prompts: list[str],
        target_latents: torch.Tensor,
        prev_states: torch.Tensor | None,
        task_embed: torch.Tensor | None,
        next_target_latents: torch.Tensor | None = None,
        sft_input_ids: torch.Tensor | None = None,
        sft_labels: torch.Tensor | None = None,
        contrastive_weight: float = 0.1,
    ) -> PipelineOutputs:
        z = self.encode(prompts, device=target_latents.device)
        state = self.step_recurrent(z, prev_states, task_embed)

        losses: dict[str, torch.Tensor] = {}

        # Stage 1: diffusion regression to encoded gold CoT latent.
        diff_loss = rectified_flow_loss(self.denoiser, target_latents, state, z, task_embed)
        losses["diffusion"] = diff_loss

        # Auxiliary anti-collapse contrastive on consecutive states (if available).
        if next_target_latents is not None and contrastive_weight > 0:
            next_state = self.step_recurrent(next_target_latents, state, task_embed)
            losses["contrastive"] = contrastive_weight * contrastive_state_loss(state, next_state)

        # Stage 2 (optional): SFT NLL on gold completion with soft-prefix.
        if sft_input_ids is not None and sft_labels is not None:
            # Use a single Euler step toward x0 to produce plan latent during training.
            plan = self.refine(z, state, task_embed, num_steps=1)
            prefix = self.make_prefix(plan)
            llm_out = self.llm.forward_with_prefix(prefix, sft_input_ids, labels=sft_labels)
            if llm_out.loss is not None:
                losses["sft"] = llm_out.loss

        return PipelineOutputs(
            plan_latent=z,
            state=state,
            losses=losses,
        )
