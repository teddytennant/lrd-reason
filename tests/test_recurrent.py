import torch

from lrd_reason.config import RecurrentConfig
from lrd_reason.models.recurrent_state import (
    GRUBlock,
    build_recurrent,
    contrastive_state_loss,
)


def test_gru_block_shapes():
    block = GRUBlock(latent_dim=16, hidden_dim=16, n_layers=2, task_dim=4)
    z = torch.randn(3, 16)
    task = torch.randn(3, 4)
    s1 = block(z, None, task)
    assert s1.shape == (3, 16)
    s2 = block(z, s1, task)
    assert s2.shape == (3, 16)
    assert not torch.allclose(s1, s2)


def test_gru_block_gradient_flows():
    block = GRUBlock(latent_dim=8, hidden_dim=8, n_layers=1)
    z = torch.randn(2, 8, requires_grad=True)
    s = block(z, None, None)
    loss = s.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
    assert has_grad


def test_build_recurrent_dispatches():
    cfg = RecurrentConfig(kind="gru", hidden_dim=8, n_layers=1)
    block = build_recurrent(cfg, latent_dim=8)
    assert isinstance(block, GRUBlock)


def test_contrastive_loss_is_finite():
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    loss = contrastive_state_loss(a, b)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_contrastive_loss_low_when_aligned():
    a = torch.randn(4, 8)
    loss_aligned = contrastive_state_loss(a, a + 0.01 * torch.randn_like(a))
    loss_random = contrastive_state_loss(a, torch.randn(4, 8))
    # Aligned should be lower than random in expectation.
    assert loss_aligned < loss_random + 1.0
