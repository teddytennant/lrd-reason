import torch

from lrd_reason.models.diffusion import (
    RectifiedFlowDenoiser,
    RFSampler,
    rectified_flow_loss,
)


def _make_model(d=8, n_layers=2):
    return RectifiedFlowDenoiser(
        latent_dim=d,
        cond_state_dim=d,
        cond_task_dim=0,
        d_model=16,
        n_layers=n_layers,
        n_heads=2,
        dropout=0.0,
    )


def test_denoiser_forward_shape():
    m = _make_model()
    x = torch.randn(4, 8)
    t = torch.rand(4)
    cond_s = torch.randn(4, 8)
    cond_z = torch.randn(4, 8)
    v = m(x, t, cond_s, cond_z)
    assert v.shape == (4, 8)


def test_rf_loss_finite_and_grad_flows():
    m = _make_model()
    x0 = torch.randn(4, 8)
    cond_s = torch.randn(4, 8)
    cond_z = torch.randn(4, 8)
    loss = rectified_flow_loss(m, x0, cond_s, cond_z)
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert has_grad


def test_sampler_returns_correct_shape():
    m = _make_model()
    sampler = RFSampler(num_steps=4)
    init = torch.randn(2, 8)
    cond_s = torch.randn(2, 8)
    cond_z = torch.randn(2, 8)
    out = sampler.sample(m, init, cond_s, cond_z)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_rf_loss_decreases_on_toy_data():
    """Train the denoiser to reproduce a fixed target set. Loss should drop.

    Sanity check: the gradient signal is alive end-to-end. Not a benchmark.
    """
    torch.manual_seed(0)
    m = _make_model(d=4, n_layers=2)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
    x0 = torch.randn(8, 4)
    cond_s = torch.zeros(8, 4)
    cond_z = torch.zeros(8, 4)

    losses = []
    for _ in range(400):
        loss = rectified_flow_loss(m, x0, cond_s, cond_z)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Last 40 should be meaningfully below first 40.
    head = sum(losses[:40]) / 40
    tail = sum(losses[-40:]) / 40
    assert tail < head * 0.75, f"head={head:.3f} tail={tail:.3f}"
