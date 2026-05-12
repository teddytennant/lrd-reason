import torch

from lrd_reason.config import AdapterConfig
from lrd_reason.models.adapter import SoftPrefixAdapter, attach_lora, count_trainable
from lrd_reason.models.llm import StubLLM


def test_soft_prefix_shape():
    adapter = SoftPrefixAdapter(latent_dim=8, n_prefix=4, llm_hidden=16)
    plan = torch.randn(3, 8)
    prefix = adapter(plan)
    assert prefix.shape == (3, 4, 16)


def test_soft_prefix_init_small():
    """Untrained prefix should be near-zero so the LLM behaves like itself initially."""
    adapter = SoftPrefixAdapter(latent_dim=16, n_prefix=2, llm_hidden=32)
    plan = torch.randn(4, 16)
    prefix = adapter(plan)
    assert prefix.abs().max().item() < 0.1


def test_attach_lora_zero_r_is_noop():
    llm = StubLLM(vocab_size=32, hidden_size=16, max_new_tokens=4)
    cfg = AdapterConfig(lora_r=0)
    out = attach_lora(llm, cfg)
    assert out is llm  # unchanged
    assert count_trainable(llm) > 0
