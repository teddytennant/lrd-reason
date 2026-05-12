import torch

from lrd_reason.config import EncoderConfig
from lrd_reason.models.encoder import EncoderProjector, StubEncoder, build_encoder


def test_stub_encoder_shape():
    enc = StubEncoder(raw_dim=64)
    out = enc.encode(["hello", "world"])
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_stub_encoder_deterministic():
    e1 = StubEncoder(raw_dim=32)
    e2 = StubEncoder(raw_dim=32)
    a = e1.encode(["abc"])
    b = e2.encode(["abc"])
    assert torch.allclose(a, b)


def test_stub_encoder_distinct_inputs_differ():
    enc = StubEncoder(raw_dim=32)
    a = enc.encode(["alpha"])
    b = enc.encode(["beta"])
    # If they collided we'd have zero distance, which is not what we want.
    assert (a - b).abs().sum().item() > 1e-3


def test_build_encoder_stub():
    cfg = EncoderConfig(kind="stub", raw_dim=16)
    enc = build_encoder(cfg)
    assert enc.raw_dim == 16


def test_projector_orthogonal_init():
    proj = EncoderProjector(raw_dim=32, latent_dim=8)
    x = torch.randn(4, 32)
    y = proj(x)
    assert y.shape == (4, 8)
    # Frozen
    assert not proj.proj.weight.requires_grad
