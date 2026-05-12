"""Config dataclasses and YAML loader.

Single source of truth for dims, losses, learning rate, model IDs. Every other module
imports from here. Ablation configs use a top-level `_inherit:` key (deep-merged onto
the parent before dataclass construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunConfig:
    name: str = "lrd-reason"
    seed: int = 0
    output_dir: str = "runs/default"
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class EncoderConfig:
    kind: str = "stub"  # stub | bge | lingbot
    hf_id: str | None = None
    raw_dim: int = 64
    pooling: str = "cls"  # cls | mean
    projector_path: str | None = None


@dataclass
class RecurrentConfig:
    kind: str = "gru"  # gru | mamba2
    hidden_dim: int = 64
    expand: int = 1
    n_layers: int = 1


@dataclass
class DiffusionConfig:
    d_model: int = 48
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.0
    n_inference_steps: int = 4


@dataclass
class AdapterConfig:
    n_prefix: int = 2
    llm_hidden: int = 32
    lora_r: int = 0
    lora_alpha: int = 0
    lora_dropout: float = 0.0
    lora_targets: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    kind: str = "stub"  # stub | hf
    hf_id: str = "stub"
    trust_remote_code: bool = True
    max_new_tokens: int = 8
    vocab_size: int = 64  # only used when kind=="stub"


@dataclass
class ModelConfig:
    latent_dim: int = 64
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@dataclass
class DataConfig:
    train_jsonl: str = ""
    val_jsonl: str = ""
    latents_path: str | None = None
    task_embed_dim: int = 16
    num_workers: int = 0


@dataclass
class TrainConfig:
    batch_size: int = 4
    micro_batch_size: int = 2
    grad_accum_steps: int = 2
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    warmup_steps: int = 0
    total_steps: int = 20
    grad_clip: float = 1.0
    log_every: int = 5
    ckpt_every: int = 100
    contrastive_weight: float = 0.1


@dataclass
class EvalConfig:
    gsm8k_n: int = 2
    math_n: int = 2
    multi_turn_scenarios: int = 2
    diffusion_step_grid: list[int] = field(default_factory=lambda: [1, 4])


@dataclass
class AblationConfig:
    use_recurrent: bool = True
    use_diffusion: bool = True
    zero_prefix: bool = False


@dataclass
class RunSpec:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self) -> None:
        ld = self.model.latent_dim
        if ld <= 0:
            raise ValueError(f"latent_dim must be > 0, got {ld}")
        if self.model.recurrent.hidden_dim != ld:
            raise ValueError(
                f"recurrent.hidden_dim ({self.model.recurrent.hidden_dim}) "
                f"must equal latent_dim ({ld})"
            )
        if self.train.grad_accum_steps * self.train.micro_batch_size != self.train.batch_size:
            raise ValueError(
                f"batch_size ({self.train.batch_size}) must equal "
                f"micro_batch_size * grad_accum_steps "
                f"({self.train.micro_batch_size} * {self.train.grad_accum_steps})"
            )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if "_inherit" in raw:
        parent_rel = raw.pop("_inherit")
        parent_path = (path.parent / parent_rel).resolve()
        parent = _load_yaml(parent_path)
        raw = _deep_merge(parent, raw)
    return raw


def _nested_dataclass_type(f) -> type | None:
    """Return the dataclass type for field `f` if it has one, else None.

    Works around `from __future__ import annotations` (which makes f.type a string)
    by probing the default_factory.
    """
    from dataclasses import MISSING, is_dataclass

    if f.default_factory is not MISSING:
        try:
            proto = f.default_factory()
        except TypeError:
            return None
        if is_dataclass(proto):
            return type(proto)
    return None


def _coerce(cls, data: dict[str, Any]):
    """Construct a dataclass from a dict, recursively coercing nested dataclass fields."""
    from dataclasses import fields

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        nested_cls = _nested_dataclass_type(f)
        if nested_cls is not None and isinstance(v, dict):
            kwargs[f.name] = _coerce(nested_cls, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def load_config(path: str | Path) -> RunSpec:
    """Load a YAML config (with _inherit support) into a validated RunSpec."""
    path = Path(path)
    raw = _load_yaml(path)
    return _coerce(RunSpec, raw)
