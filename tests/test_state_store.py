import tempfile
from pathlib import Path

import torch

from lrd_reason.infer.state_store import delete_state, load_state, save_state


def test_save_load_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        s = torch.randn(16)
        save_state(tmp, "abc-123", s, turn_idx=3, meta={"hello": "world"})
        loaded = load_state(tmp, "abc-123")
        assert loaded is not None
        assert torch.allclose(loaded.state, s)
        assert loaded.turn_idx == 3
        assert loaded.meta == {"hello": "world"}


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_state(tmp, "nope") is None


def test_delete_state():
    with tempfile.TemporaryDirectory() as tmp:
        save_state(tmp, "to-delete", torch.zeros(4), turn_idx=0)
        assert delete_state(tmp, "to-delete") is True
        assert load_state(tmp, "to-delete") is None
        assert delete_state(tmp, "to-delete") is False


def test_atomic_write_no_partial_file():
    with tempfile.TemporaryDirectory() as tmp:
        s = torch.randn(8)
        save_state(tmp, "x", s, turn_idx=0)
        # No leftover .tmp files
        tmps = list(Path(tmp).glob("*.tmp"))
        assert tmps == []
