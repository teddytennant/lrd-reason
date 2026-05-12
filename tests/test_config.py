from pathlib import Path

from lrd_reason.config import RunSpec, load_config

REPO = Path(__file__).resolve().parents[1]


def test_smoke_config_loads(smoke_spec: RunSpec):
    assert smoke_spec.model.latent_dim == 64
    assert smoke_spec.run.device == "cpu"
    assert smoke_spec.train.total_steps == 20


def test_main_config_loads():
    spec = load_config(REPO / "configs" / "main.yaml")
    assert spec.model.latent_dim == 256
    assert spec.model.encoder.kind == "bge"
    assert spec.model.recurrent.kind == "mamba2"


def test_ablation_inheritance():
    base = load_config(REPO / "configs" / "main.yaml")
    full = load_config(REPO / "configs" / "ablations" / "full.yaml")
    # _inherit should carry over the model config from main
    assert full.model.latent_dim == base.model.latent_dim
    assert full.ablation.use_recurrent is True
    assert full.ablation.use_diffusion is True

    baseline = load_config(REPO / "configs" / "ablations" / "baseline.yaml")
    assert baseline.ablation.use_recurrent is False
    assert baseline.ablation.use_diffusion is False
    assert baseline.ablation.zero_prefix is True


def test_invalid_dims_rejected():
    import pytest

    from lrd_reason.config import RecurrentConfig, RunSpec

    with pytest.raises(ValueError):
        # latent_dim != recurrent.hidden_dim
        spec = RunSpec()
        spec.model.latent_dim = 64
        spec.model.recurrent = RecurrentConfig(hidden_dim=128)
        spec.__post_init__()
