"""Shared training infrastructure.

Atomic checkpoint writes, RNG capture, optimizer/dataloader-step recovery, resume
from `checkpoints/{phase}/latest.symlink`. Single-process here; FSDP wrap happens
in the per-stage entry points on the H200 box.

Key invariant: every checkpoint contains everything needed to resume bit-for-bit.
LAUNCH.md's stop/resume drill exercises this.
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import RunSpec


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(obj: Any, dest: Path) -> None:
    """Write to a tmp file in the same directory, then rename. Safe on SIGTERM."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        torch.save(obj, tmp_path)
    os.replace(tmp_path, dest)


def _update_latest_symlink(ckpt_dir: Path, ckpt_path: Path) -> None:
    link = ckpt_dir / "latest.symlink"
    tmp = ckpt_dir / "latest.symlink.tmp"
    if tmp.exists():
        tmp.unlink()
    tmp.symlink_to(ckpt_path.name)
    os.replace(tmp, link)


def save_checkpoint(
    ckpt_dir: str | Path,
    state: TrainState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    extra: dict[str, Any] | None = None,
) -> Path:
    ckpt_dir = Path(ckpt_dir)
    payload: dict[str, Any] = {
        "step": state.step,
        "epoch": state.epoch,
        "best_loss": state.best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    out = ckpt_dir / f"step_{state.step:08d}.pt"
    _atomic_torch_save(payload, out)
    _update_latest_symlink(ckpt_dir, out)
    return out


def load_checkpoint(
    ckpt_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainState | None:
    ckpt_dir = Path(ckpt_dir)
    link = ckpt_dir / "latest.symlink"
    if not link.exists():
        return None
    target = (ckpt_dir / os.readlink(link)).resolve()
    payload = torch.load(target, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return TrainState(
        step=payload.get("step", 0),
        epoch=payload.get("epoch", 0),
        best_loss=payload.get("best_loss", float("inf")),
    )


def build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """AdamW over all trainable params. Returns even if model has no trainable params
    (dummy 1-param group so optimizer state-dict round-trip works)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        params = [torch.nn.Parameter(torch.zeros(1))]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    return base_lr * (step + 1) / warmup_steps


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def output_dirs(spec: RunSpec, phase: str) -> tuple[Path, Path]:
    """Returns (run_root, ckpt_dir)."""
    root = Path(spec.run.output_dir)
    ckpt = root / "checkpoints" / phase
    ckpt.mkdir(parents=True, exist_ok=True)
    return root, ckpt
