"""Curriculum Stage 1: unsupervised continued pretraining via the LoRA adapter.

Runs BEFORE the cold-start stages (`stage1.py` + `stage2.py`). The goal is to
imprint reasoning patterns and voice from real human truth-seekers rather than
from a synthetic teacher whose hedging would otherwise be baked into the model.

Data: raw text from `data.pretrain_corpus` (files/globs), packed into
fixed-length blocks of `data.pretrain_block_size` tokens by
`data/text_dataset.py`. The intended corpus mix is documented in README
"Training curriculum": complete Nietzsche + Kant (Gutenberg), Lean mathlib4,
arXiv math, SQLite/TigerBeetle/CPython source, FineWeb-Edu slice.

Loss: standard next-token cross-entropy on packed blocks. The soft prefix is
all-zeros (no plan latent exists yet), so the LLM trains as an ordinary causal
LM. No generator targets, no CoT format at this stage.

Trainable: only the LoRA params peft attached to the frozen HF LLM. Recurrent
state, denoiser, adapter, and projector are frozen here; they enter training in
the cold-start stages. On the stub path (lora_r == 0, no peft) the StubLLM
itself trains, which keeps the CPU smoke test a real optimization problem.

Checkpoints under `checkpoints/pretrain/` via `loop.py` (atomic, resumable).
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import torch
from torch.utils.data import DataLoader

from ..config import RunSpec, load_config
from ..data.text_dataset import PackedTextDataset, byte_encode, resolve_corpus_paths
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

PHASE = "pretrain"


def _encode_fn(llm: torch.nn.Module, vocab_fallback: int) -> Callable[[str], list[int]]:
    """HF path: real tokenizer stream. Stub path: byte-level ids mod vocab."""
    if hasattr(llm, "encode_text"):
        return llm.encode_text  # type: ignore[no-any-return]
    vocab_size = int(getattr(llm, "vocab_size", vocab_fallback))
    return lambda text: byte_encode(text, vocab_size)


def _build_dataloader(spec: RunSpec, llm: torch.nn.Module) -> DataLoader:
    if not spec.data.pretrain_corpus:
        raise ValueError(
            "data.pretrain_corpus is empty — point it at raw .txt files/globs "
            "(see configs/smoke.yaml for the smoke corpus)"
        )
    paths = resolve_corpus_paths(spec.data.pretrain_corpus)
    ds = PackedTextDataset(
        paths=paths,
        block_size=spec.data.pretrain_block_size,
        encode_fn=_encode_fn(llm, vocab_fallback=spec.model.llm.vocab_size),
    )
    return DataLoader(
        ds,
        batch_size=spec.train.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=spec.data.num_workers,
    )


def _freeze_non_llm(pipeline: LRDPipeline) -> None:
    """Only the LLM's trainable params (LoRA, or the whole StubLLM) may train."""
    for module in (pipeline.recurrent, pipeline.denoiser, pipeline.adapter, pipeline.projector):
        for p in module.parameters():
            p.requires_grad = False


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
    _freeze_non_llm(pipeline)

    n_trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError(
            "Stage 1 pretrain has nothing to train: LLM is frozen and no LoRA is "
            "attached. Set adapter.lora_r > 0 (HF path) or use llm.kind=stub."
        )
    print(f"[{PHASE}] trainable params: {n_trainable:,}")

    optimizer = build_optimizer(pipeline, lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    _, ckpt_dir = output_dirs(spec, phase=PHASE)

    state = TrainState()
    if resume:
        loaded = load_checkpoint(ckpt_dir, pipeline, optimizer)
        if loaded is not None:
            state = loaded

    loader_iter = _make_iter(_build_dataloader(spec, pipeline.llm))
    total_steps = max_steps if max_steps is not None else spec.train.total_steps

    # Zero soft prefix: the LLM trains as a plain causal LM at this stage.
    n_prefix = max(1, pipeline.adapter.n_prefix)
    llm_hidden = pipeline.adapter.llm_hidden

    first_loss = float("nan")
    last_loss = float("nan")
    while state.step < total_steps:
        ids = next(loader_iter).to(device)
        embed_dtype = pipeline.llm.embed(ids[:1, :1]).dtype
        prefix = torch.zeros(
            ids.shape[0], n_prefix, llm_hidden, dtype=embed_dtype, device=device
        )

        out = pipeline.llm.forward_with_prefix(prefix, ids, labels=ids)
        if out.loss is None:
            raise RuntimeError("LLM returned no loss despite labels being provided")
        loss = out.loss / spec.train.grad_accum_steps
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
        if math.isnan(first_loss):
            first_loss = last_loss
        if state.step % spec.train.log_every == 0:
            print(f"[{PHASE}] step={state.step} nll={last_loss:.4f}")
        if state.step % spec.train.ckpt_every == 0:
            save_checkpoint(ckpt_dir, state, pipeline, optimizer)

    save_checkpoint(ckpt_dir, state, pipeline, optimizer)
    return {"first_loss": first_loss, "final_loss": last_loss, "steps": state.step}


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
