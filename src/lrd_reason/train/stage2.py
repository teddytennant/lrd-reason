"""Stage 2 of the cold-start: adapter + LoRA SFT on gold completions, with the refined plan latent as prefix.

Together with `stage1.py`, completes the cold-start phase of the three-stage
curriculum. Stage 1 (`stage1.py`) must have produced a checkpoint (or be loaded
via --stage1-checkpoint). Stage 2 thaws the adapter and (if configured) attaches
LoRA on the LLM. After this finishes, the resulting checkpoint is the anchor that
Stage 3 (`stage_rlvr.py`) does RL on top of, with a KL penalty against this
checkpoint to prevent the policy from drifting away from the cold-start posture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def _load_stage1(pipeline: LRDPipeline, stage1_path: Path) -> None:
    if not stage1_path.exists():
        # Allow symlink resolution
        if stage1_path.parent.exists() and (stage1_path.parent / "latest.symlink").exists():
            stage1_path = stage1_path.parent / (stage1_path.parent / "latest.symlink").readlink()
        else:
            raise FileNotFoundError(f"stage1 checkpoint not found: {stage1_path}")
    payload = torch.load(stage1_path, map_location="cpu", weights_only=False)
    pipeline.load_state_dict(payload["model"], strict=False)


def _build_sft_dataloader(spec: RunSpec) -> DataLoader:
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


def _make_sft_batch(
    pipeline: LRDPipeline, batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize gold_cot+gold_answer to (input_ids, labels). Stub LLM uses a hash
    fallback so this path is CPU-runnable."""
    completions = [
        (cot + ("\n" if cot else "") + ans).strip() or "."
        for cot, ans in zip(batch["gold_cots"], batch["gold_answers"], strict=True)
    ]
    llm = pipeline.llm
    if hasattr(llm, "tokenize"):
        ids = llm.tokenize(completions).to(device)  # type: ignore[attr-defined]
    else:
        # Stub path: deterministic toy tokenization (char ords mod vocab).
        max_len = 8
        vocab_size = getattr(llm, "vocab_size", 64)
        ids = torch.zeros(len(completions), max_len, dtype=torch.long, device=device)
        for i, s in enumerate(completions):
            for j, ch in enumerate(s[:max_len]):
                ids[i, j] = ord(ch) % vocab_size
    labels = ids.clone()
    return ids, labels


def train(
    spec: RunSpec,
    resume: bool,
    stage1_checkpoint: str | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    set_seed(spec.run.seed)
    device = torch.device(spec.run.device)

    pipeline = LRDPipeline(
        model_cfg=spec.model,
        ablation=spec.ablation,
        task_dim=spec.data.task_embed_dim,
        device=spec.run.device,
        dtype=getattr(torch, spec.run.dtype),
    ).to(device)

    if stage1_checkpoint is not None:
        _load_stage1(pipeline, Path(stage1_checkpoint))

    # Stage 2: freeze recurrent + diffusion (already trained); train adapter +
    # whatever LoRA params peft created on the LLM.
    for p in pipeline.recurrent.parameters():
        p.requires_grad = False
    for p in pipeline.denoiser.parameters():
        p.requires_grad = False
    for p in pipeline.adapter.parameters():
        p.requires_grad = True
    # LoRA params (if any) keep their peft default (trainable). Frozen base LLM
    # params remain frozen by peft.

    optimizer = build_optimizer(pipeline, lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    _, ckpt_dir = output_dirs(spec, phase="stage2")

    state = TrainState()
    if resume:
        loaded = load_checkpoint(ckpt_dir, pipeline, optimizer)
        if loaded is not None:
            state = loaded

    loader_iter = _make_iter(_build_sft_dataloader(spec))
    total_steps = max_steps if max_steps is not None else spec.train.total_steps

    last_loss = float("nan")
    while state.step < total_steps:
        batch = next(loader_iter)
        targets = batch["target_latents"].to(device)
        task_embeds = batch["task_embeds"].to(device) if spec.data.task_embed_dim > 0 else None
        sft_ids, sft_labels = _make_sft_batch(pipeline, batch, device)

        out = pipeline.training_step(
            prompts=batch["prompts"],
            target_latents=targets,
            prev_states=None,
            task_embed=task_embeds,
            sft_input_ids=sft_ids,
            sft_labels=sft_labels,
            contrastive_weight=0.0,
        )
        if "sft" not in out.losses:  # type: ignore[union-attr]
            raise RuntimeError("Stage 2 expected an SFT loss but pipeline did not produce one")
        loss = out.losses["sft"] / spec.train.grad_accum_steps  # type: ignore[union-attr]
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
            print(f"[stage2] step={state.step} sft_loss={last_loss:.4f}")
        if state.step % spec.train.ckpt_every == 0:
            save_checkpoint(ckpt_dir, state, pipeline, optimizer)

    save_checkpoint(ckpt_dir, state, pipeline, optimizer)
    return {"final_loss": last_loss, "steps": state.step}


def _make_iter(loader: DataLoader):
    while True:
        yield from loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--stage1-checkpoint", type=str, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    spec = load_config(args.config)
    train(
        spec,
        resume=args.resume,
        stage1_checkpoint=args.stage1_checkpoint,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
