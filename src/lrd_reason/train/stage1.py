"""Stage 1: joint training of recurrent state + rectified-flow diffusion denoiser.

Loss = rectified_flow_loss + contrastive_weight * contrastive_state_loss.

The frozen encoder and LLM are not used here (saves an HF forward per batch).
We train only on (prompt -> target_latent) pairs from the data pipeline.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..config import RunSpec, load_config
from ..data.collate import collate_pairs
from ..data.dataset import LatentPairDataset
from ..models.pipeline import LRDPipeline
from .loop import (
    TrainState,
    build_optimizer,
    load_checkpoint,
    output_dirs,
    save_checkpoint,
    set_lr,
    set_seed,
    warmup_lr,
)


def _build_dataloader(spec: RunSpec) -> DataLoader:
    ds = LatentPairDataset(
        jsonl_path=spec.data.train_jsonl,
        latent_dim=spec.model.latent_dim,
        latents_path=spec.data.latents_path,
        task_embed_dim=spec.data.task_embed_dim,
    )
    return DataLoader(
        ds,
        batch_size=spec.train.micro_batch_size,
        collate_fn=collate_pairs,
        num_workers=spec.data.num_workers,
    )


def train(spec: RunSpec, resume: bool, max_steps: int | None = None) -> dict[str, float]:
    set_seed(spec.run.seed)
    device = torch.device(spec.run.device)

    pipeline = LRDPipeline(
        model_cfg=spec.model,
        ablation=spec.ablation,
        task_dim=spec.data.task_embed_dim,
        device=spec.run.device,
        dtype=getattr(torch, spec.run.dtype),
    ).to(device)

    # Stage 1 only trains recurrent + diffusion denoiser; freeze adapter + LLM.
    for p in pipeline.adapter.parameters():
        p.requires_grad = False
    for p in pipeline.llm.parameters():
        p.requires_grad = False

    optimizer = build_optimizer(pipeline, lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    _, ckpt_dir = output_dirs(spec, phase="stage1")

    state = TrainState()
    if resume:
        loaded = load_checkpoint(ckpt_dir, pipeline, optimizer)
        if loaded is not None:
            state = loaded

    loader_iter = _make_iter(_build_dataloader(spec))
    total_steps = max_steps if max_steps is not None else spec.train.total_steps

    last_loss = float("nan")
    while state.step < total_steps:
        batch = next(loader_iter)
        targets = batch["target_latents"].to(device)
        task_embeds = batch["task_embeds"].to(device) if spec.data.task_embed_dim > 0 else None

        # Construct a "next-turn" latent for the contrastive loss by rolling within
        # the batch. Cheap, doesn't require real multi-turn data for the smoke path.
        next_targets = torch.roll(targets, shifts=-1, dims=0)

        out = pipeline.training_step(
            prompts=batch["prompts"],
            target_latents=targets,
            prev_states=None,
            task_embed=task_embeds,
            next_target_latents=next_targets,
            contrastive_weight=spec.train.contrastive_weight,
        )
        loss = sum(out.losses.values())  # type: ignore[union-attr]
        loss = loss / spec.train.grad_accum_steps
        loss.backward()

        if (state.step + 1) % spec.train.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in pipeline.parameters() if p.requires_grad],
                max_norm=spec.train.grad_clip,
            )
            lr = warmup_lr(state.step, spec.train.warmup_steps, spec.train.lr)
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        state.step += 1
        last_loss = float(loss.detach().item()) * spec.train.grad_accum_steps
        if state.step % spec.train.log_every == 0:
            print(f"[stage1] step={state.step} loss={last_loss:.4f}")
        if state.step % spec.train.ckpt_every == 0:
            save_checkpoint(ckpt_dir, state, pipeline, optimizer)

    save_checkpoint(ckpt_dir, state, pipeline, optimizer)
    return {"final_loss": last_loss, "steps": state.step}


def _make_iter(loader: DataLoader):
    """Infinite iterator over a (finite) DataLoader."""
    while True:
        yield from loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    spec = load_config(args.config)
    train(spec, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
