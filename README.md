# lrd-reason

Latent Recurrent Diffusion Reasoning hybrid. Bolts a persistent recurrent state and a latent rectified-flow refinement module onto a frozen instruction-tuned LLM. The LLM stays frozen except for a small LoRA adapter; reasoning happens in a 256-dim latent space before generation.

This repo is the **code build** — runnable end-to-end on CPU via stubs, ready to ship to 2× H200 when training is funded. No weights are downloaded and no training is executed by `pytest`.

## Project Values

This codebase is governed by the Constitution for Truth-Seeking AI. The full 7-principle constitution lives in `src/lrd_reason/constitution.py` as `FULL_CONSTITUTION`, alongside two distilled prompts:

- `COT_SYSTEM_PROMPT` — the per-example system prompt, used both for Stage-2 cold-start trace generation and as the inference-time policy shaper for the deployed model. Encodes Occam, verify-before-answer, bold conjecture + ruthless self-attack (the Dionysian/Apollonian synthesis), independent thinking, epistemic honesty, no padding.
- `PROBLEM_GENERATOR_PROMPT` — the system prompt for synthesising fresh verifiable problems where no suitable open dataset exists.

The full constitution:

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

## Training curriculum

Three stages, designed so that values come from real human truth-seekers (raw text) rather than synthetic teacher imitation, and behaviour comes from verifier-grounded RL rather than imitation of CoT traces.

| Stage | Goal | What trains | Data | Status |
|---|---|---|---|---|
| **1. Unsupervised pretrain** | Imprint voice and reasoning patterns from real exemplars | LoRA adapter on the LLM | Raw text: complete Nietzsche + Kant from Project Gutenberg, Lean mathlib4, arXiv math, SQLite/TigerBeetle/CPython source, FineWeb-Edu slice | TO BUILD (`train/stage_pretrain.py`) |
| **2. Cold-start latent SFT** | Seed the recurrent + diffusion modules with the `(prompt → latent → CoT)` mapping | Recurrent + diffusion; then adapter + LoRA | ~10–50k high-quality CoT traces from Stage-2 generator (`scripts/generate_data.py`) under `COT_SYSTEM_PROMPT` | Built (`train/stage1.py` + `train/stage2.py`) |
| **3. RLVR main phase** | Grow real reasoning competence from verifier reward | All trainable modules, with KL anchor to Stage-2 checkpoint | Verifiable problems: MATH, GSM8K, CodeContests, APPS, BBH, LogiQA, mathlib4 type-checks | TO BUILD (`train/stage_rlvr.py` + `eval/verifiers/`) |

Philosophy is **Stage 1 only** (raw text → LoRA voice imprint). No synthetic Q&A is generated from Kant or Nietzsche; the stance only transfers from the authors themselves.

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

## Side track: 4B uncensored LoRA SFT

A shallow, deployable companion to the main 35B latent-reasoning system. LoRA fine-tunes `HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive` on the same data philosophy — Kant + Nietzsche raw text for voice, `COT_SYSTEM_PROMPT` traces for reasoning — but with no recurrent or diffusion modules. The uncensored base is chosen so principle VII (Truth Over Obedience) doesn't have to fight a refusal prior.

```bash
# 1. CoT traces (reuse the existing Stage-2 generator).
python scripts/generate_data.py --config configs/main.yaml \
    --problems data/problems.jsonl --output data/cot.uncensored.jsonl

# 2. Mix philosophy + CoT into one SFT corpus.
python scripts/prepare_uncensored_data.py \
    --philosophy data/philosophy/ \
    --cot data/cot.uncensored.jsonl \
    --output data/uncensored_sft.jsonl --mix 0.35

# 3. LoRA SFT (~16GB VRAM bf16, ~10GB with quantize_4bit: true).
python scripts/finetune_uncensored.py --config configs/finetune_uncensored.yaml
```

## Layout

```
configs/        YAML configs (main, smoke, ablations/{baseline,recurrent_only,diffusion_only,full})
src/lrd_reason/
  config.py         dataclass configs + YAML loader
  constitution.py   distilled CoT + problem-generator prompts; FULL_CONSTITUTION
  models/           encoder, recurrent_state, diffusion, adapter, pipeline
  data/             dataset, collate, cot_generator
  train/            loop, stage1, stage2 (cold-start); stage_pretrain, stage_rlvr (to build)
  infer/            state_store, pipeline, cli
  eval/             gsm8k, math_bench, multi_turn, metrics, ablations; verifiers/ (to build)
scripts/        thin entry points
tests/          pytest suite (CPU-only)
```
