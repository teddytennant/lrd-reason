# LAUNCH — 48-Hour H200 Runbook (segmentable)

Deferred-execution doc. Nothing here runs as part of `pytest`. Read top-to-bottom before spending money.

## Budget

- **Hard walltime cap:** 48h continuous, or any combination of shorter sessions totaling ≤48h.
- **Cost:** 2× H200 @ ~$8/hr × 48h ≈ **$770**.
- **Cheaper option:** 1× H200 for most of training (FSDP-1 = DDP-of-one); 2× H200 only for the data-generation burst. Cuts cost roughly in half.

This budget is tighter than the original spec's 72h. The schedule below is sized so each phase can stop and resume from a checkpoint, with no phase exceeding the smallest plausible single-session budget (24h).

## Segmentation contract

Every phase must:
- Save complete resumable state (model + optimizer + RNG + dataloader step + RF timestep) to `checkpoints/{phase}/{step}.pt` at least every 4h.
- Write `checkpoints/{phase}/latest.symlink` pointing to the most recent checkpoint.
- Have a `--resume` flag that picks up from `latest.symlink` if present.
- Be killable by SIGTERM without corrupting the last checkpoint (write atomically: tmp + rename).

This is enforced by `src/lrd_reason/train/loop.py`; do not bypass it.

## Hardware assumption

2× NVIDIA H200 (141GB each), CUDA 12.4+, NVLink between cards, ≥1.5TB local NVMe.

## Phase 0 — Setup (≤2h, one-time per machine)

```bash
sudo apt update && sudo apt install -y git tmux htop nvtop
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11
. .venv/bin/activate
uv pip install -e ".[hf,vllm,mamba,dev]"
python scripts/download_models.py     # ~80GB, one-time
pytest -q tests/                       # sanity
nvidia-smi
```

If Phase 0 is split across reboots, re-source `.venv/bin/activate` before each subsequent phase.

## Phase 1 — Data generation (≤10h on 2× H200)

```bash
# Both GPUs in parallel via tmux. Each shard is independently resumable.
CUDA_VISIBLE_DEVICES=0 python scripts/generate_data.py \
    --config configs/main.yaml --output data/cot.0.jsonl \
    --shard 0 --num-shards 2 --target 60000 --resume

CUDA_VISIBLE_DEVICES=1 python scripts/generate_data.py \
    --config configs/main.yaml --output data/cot.1.jsonl \
    --shard 1 --num-shards 2 --target 60000 --resume

# After both finish:
python scripts/encode_targets.py --inputs data/cot.0.jsonl data/cot.1.jsonl --out data/latents.pt
```

Expected: ~120k examples, ~8–10h on 2× H200 with Qwen3.5-35B-A3B-FP8 + vLLM. If throughput is <40 ex/min/GPU after 30 min of warmup, kill and tune vLLM batch size before continuing.

## Phase 2 — Stage 1 training (≤22h)

Recurrent + diffusion joint training. Resumable.

```bash
torchrun --nproc-per-node=2 -m lrd_reason.train.stage1 \
    --config configs/main.yaml --resume
```

Target: 1.5–2.5 epochs over 120k examples. Checkpoint every 2000 steps (~2h).

**Stop/resume drill (do this once on a small slice before committing):** train for 15 min, kill with SIGTERM, restart with `--resume`, confirm loss curve continues without restart.

## Phase 3 — Stage 2 training (≤8h)

Adapter + LoRA SFT.

```bash
torchrun --nproc-per-node=2 -m lrd_reason.train.stage2 \
    --config configs/main.yaml \
    --stage1-checkpoint checkpoints/stage1/latest.pt \
    --resume
```

## Phase 4 — Ablation eval (≤5h)

```bash
python scripts/run_eval.py --config configs/main.yaml --out results/ --resume
```

Runs baseline / recurrent_only / diffusion_only / full and emits `results/ablations.md`. Resumable per-config — partial results are written incrementally.

## Phase 5 — Demo + docs (≤1h)

```bash
python scripts/demo.py --config configs/main.yaml \
    --prompts "A train leaves Chicago at 3pm…" "Now what if it leaves at 5pm?"
git add results/ && git commit -m "results: ablation table" && git push
```

## Total budget

| Phase | Cap | Cumulative |
|---|---|---|
| 0  | 2h  | 2h  |
| 1  | 10h | 12h |
| 2  | 22h | 34h |
| 3  | 8h  | 42h |
| 4  | 5h  | 47h |
| 5  | 1h  | 48h |

If running in segments, recommended split points are between phases (cleanest resume contract). Mid-phase stops also work but require the `--resume` drill to pass first.

## Result template

| Configuration | GSM8K hard | MATH | Multi-turn consistency | Self-correction | Latency p50 |
|---|---|---|---|---|---|
| Baseline (Qwen + zero prefix) | | | | | |
| + Recurrent only | | | | | |
| + Diffusion only (K=8) | | | | | |
| Full (K=4) | | | | | |
| Full (K=8) | | | | | |
| Full (K=16) | | | | | |

**Success criterion:** Full system improves monotonically with K (4 → 8 → 16); ablations do not.

## Trim list (if running short on time)

In priority order, drop these first:
1. K=16 ablation row (keep 1/4/8).
2. MATH eval (keep GSM8K + multi-turn).
3. Self-correction test (keep consistency).
4. Drop Stage 2 LoRA, keep only adapter+soft-prefix. (Hurts headline result; only if very behind.)

## Kill switches

- VRAM OOM during data gen → drop vLLM `max_num_batched_tokens` to 16384, retry.
- Stage 1 loss diverges → restart from last checkpoint with lr halved, grad-clip 0.5.
- Generation slower than 40 ex/min/GPU after 30-min warmup → kill, halve batch, restart.
- Total walltime burn rate exceeding the cap → stop after current phase, segment the rest into a second session.
