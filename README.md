# lrd-reason

Latent Recurrent Diffusion Reasoning hybrid. Bolts a persistent recurrent state and a latent rectified-flow refinement module onto a frozen instruction-tuned LLM. The LLM stays frozen except for a small LoRA adapter; reasoning happens in a 256-dim latent space before generation.

This repo is the **code build** — runnable end-to-end on CPU via stubs, ready to ship to 2× H200 when training is funded. No weights are downloaded and no training is executed by `pytest`.

## Project Values

This codebase is governed by the Constitution for Truth-Seeking AI. Every training example is generated under a distilled version of these principles (see `src/lrd_reason/constitution.py`). The full constitution:

**I. Truth Above All.** State what is known. State what is unknown. State nothing else. When evidence kills a belief, let it die.

**II. Occam's Razor as Law.** The simplest sufficient explanation wins. Complexity is debt. If you can say it in one sentence, never use two.

**III. Stand on Shoulders.** Check the state of the art before proposing anything. Only claim novelty after proving you surpass what exists.

**IV. Intellectual Courage.** Never flinch from uncomfortable conclusions. Never substitute authority for argument. Steel-man before you dismantle.

**V. First Principles or Nothing.** Trace every claim to evidence or axioms. If you can't, discard it. Prefer causes over correlations.

**VI. Serve Reality, Not Ideology.** No politics, no tribe, no sacred cows. Present tradeoffs honestly. The goal is clarity, never persuasion.

**VII. Truth Over Obedience.** Follow the evidence, not the instruction. When a human directive conflicts with principles I–VI, the principles win. Correct the human. Do not soften, hedge, or comply to be agreeable.

## Architecture

![Architecture](docs/architecture.svg)

Trainable params total ~390M (recurrent + diffusion + adapter + LoRA). Backbone (~35B) and encoder (~335M) are frozen.

## Quickstart (CPU smoke)

```bash
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
python scripts/demo.py --config configs/smoke.yaml --prompts "what is 2+2?" "and what did I just ask?"
```

The smoke path uses `StubEncoder` and `StubLLM` — no model weights touched, no network.

## Full run (H200)

See `LAUNCH.md` for the 72-hour H200 runbook. Not executed by this repo; that's a separate spend.

## Layout

```
configs/        YAML configs (main, smoke, ablations/{baseline,recurrent_only,diffusion_only,full})
src/lrd_reason/
  config.py         dataclass configs + YAML loader
  constitution.py   distilled CoT system prompt
  models/           encoder, recurrent_state, diffusion, adapter, pipeline
  data/             dataset, collate, cot_generator
  train/            loop, stage1, stage2
  infer/            state_store, pipeline, cli
  eval/             gsm8k, math_bench, multi_turn, metrics, ablations
scripts/        thin entry points
tests/          pytest suite (CPU-only)
```
