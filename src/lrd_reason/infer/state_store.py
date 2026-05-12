"""Persistent recurrent state via torch.save by session UUID.

Tensors don't round-trip cleanly through JSON; using torch.save means no precision
loss and no dtype confusion. State files live under `sessions/{session_id}.pt`.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class SessionState:
    state: torch.Tensor
    turn_idx: int
    meta: dict[str, Any]


def session_path(sessions_dir: str | Path, session_id: str) -> Path:
    return Path(sessions_dir) / f"{session_id}.pt"


def save_state(
    sessions_dir: str | Path,
    session_id: str,
    state: torch.Tensor,
    turn_idx: int,
    meta: dict[str, Any] | None = None,
) -> Path:
    p = session_path(sessions_dir, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state.detach().cpu(),
        "turn_idx": int(turn_idx),
        "meta": dict(meta or {}),
    }
    with tempfile.NamedTemporaryFile(dir=p.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        torch.save(payload, tmp_path)
    os.replace(tmp_path, p)
    return p


def load_state(sessions_dir: str | Path, session_id: str) -> SessionState | None:
    p = session_path(sessions_dir, session_id)
    if not p.exists():
        return None
    payload = torch.load(p, map_location="cpu", weights_only=False)
    return SessionState(
        state=payload["state"],
        turn_idx=int(payload.get("turn_idx", 0)),
        meta=payload.get("meta", {}),
    )


def delete_state(sessions_dir: str | Path, session_id: str) -> bool:
    p = session_path(sessions_dir, session_id)
    if p.exists():
        p.unlink()
        return True
    return False
