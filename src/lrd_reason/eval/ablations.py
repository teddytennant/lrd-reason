"""Ablation runner.

For each of {baseline, recurrent_only, diffusion_only, full}, build an
InferenceRunner, sweep diffusion step counts where applicable, and emit a
markdown table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import RunSpec, load_config
from ..infer.pipeline import InferenceRunner
from . import gsm8k, math_bench, multi_turn

ABLATION_CONFIGS = [
    ("baseline", "configs/ablations/baseline.yaml"),
    ("recurrent_only", "configs/ablations/recurrent_only.yaml"),
    ("diffusion_only", "configs/ablations/diffusion_only.yaml"),
    ("full", "configs/ablations/full.yaml"),
]


@dataclass
class RowResult:
    ablation: str
    diffusion_steps: int
    gsm8k_acc: float
    math_acc: float
    multi_turn_consistency: float
    contradiction_rate: float


def evaluate_one(
    spec: RunSpec,
    stage1_ckpt: str | None,
    stage2_ckpt: str | None,
    diffusion_steps: int,
) -> dict[str, Any]:
    runner = InferenceRunner(spec=spec, stage1_ckpt=stage1_ckpt, stage2_ckpt=stage2_ckpt)
    g = gsm8k.evaluate(runner, n=spec.eval.gsm8k_n, num_diffusion_steps=diffusion_steps)
    m = math_bench.evaluate(runner, n=spec.eval.math_n, num_diffusion_steps=diffusion_steps)
    mt = multi_turn.evaluate(
        runner,
        n_scenarios=spec.eval.multi_turn_scenarios,
        num_diffusion_steps=diffusion_steps,
    )
    return {
        "gsm8k_acc": g["accuracy"],
        "math_acc": m["accuracy"],
        "multi_turn_consistency": mt["consistency"],
        "contradiction_rate": mt["contradiction_rate"],
    }


def run(
    base_config_path: str | Path,
    out_path: str | Path,
    stage1_ckpt: str | None = None,
    stage2_ckpt: str | None = None,
) -> list[RowResult]:
    base_dir = Path(base_config_path).parent
    rows: list[RowResult] = []
    base = load_config(base_config_path)
    steps_grid = base.eval.diffusion_step_grid

    for name, rel in ABLATION_CONFIGS:
        cfg_path = Path(rel) if Path(rel).is_absolute() else (base_dir.parent / rel)
        if not cfg_path.exists():
            # Allow running directly from configs/ paths supplied verbatim.
            cfg_path = Path(rel)
        spec = load_config(cfg_path)
        # If diffusion is disabled by ablation, only run with steps=1.
        grid = steps_grid if spec.ablation.use_diffusion else [1]
        for k in grid:
            scores = evaluate_one(spec, stage1_ckpt, stage2_ckpt, diffusion_steps=k)
            rows.append(RowResult(ablation=name, diffusion_steps=k, **scores))

    _write_markdown(out_path, rows)
    return rows


def _write_markdown(path: str | Path, rows: list[RowResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Ablation", "K", "GSM8K", "MATH", "Multi-turn", "Contradiction"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        d = asdict(r)
        lines.append(
            f"| {d['ablation']} | {d['diffusion_steps']} | "
            f"{d['gsm8k_acc']:.3f} | {d['math_acc']:.3f} | "
            f"{d['multi_turn_consistency']:.3f} | {d['contradiction_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")
