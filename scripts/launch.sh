#!/usr/bin/env bash
# One-liner driver for the uncensored 4B LoRA SFT track (philosophy-only).
#
# Curl-pipe-bash usage (recommended):
#   curl -sL https://raw.githubusercontent.com/teddytennant/lrd-reason/qwen-uncensored-finetune/scripts/launch.sh | bash
#
# Local usage (already cloned, in repo root):
#   ./scripts/launch.sh
#
# Env vars:
#   LRD_WORKDIR  — where to clone (default: $HOME/lrd-uncensored-run). Ignored if
#                  the script is run from inside an existing checkout.
#   LRD_BRANCH   — branch to check out (default: qwen-uncensored-finetune).
#   LRD_NO_TUI=1 — skip the TUI; trainer logs stream to stdout as usual.
#   LRD_DRY_RUN=1— set up env + data, skip the actual train command.

set -euo pipefail

REPO_URL="https://github.com/teddytennant/lrd-reason.git"
BRANCH="${LRD_BRANCH:-qwen-uncensored-finetune}"
WORKDIR="${LRD_WORKDIR:-$HOME/lrd-uncensored-run}"

c_blue() { printf "\033[1;34m%s\033[0m\n" "$*"; }
c_green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
c_yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }
c_red() { printf "\033[1;31m%s\033[0m\n" "$*" >&2; }

step() { c_blue "==> $*"; }

# 0. Locate / clone the repo.
DID_CLONE_OR_FETCH=0
if [ -f "scripts/finetune_uncensored.py" ] && [ -f "configs/finetune_uncensored.yaml" ]; then
  step "using existing checkout at $(pwd)"
else
  step "cloning $REPO_URL ($BRANCH) -> $WORKDIR"
  if [ -d "$WORKDIR/.git" ]; then
    c_yellow "  workdir already exists, fetching latest"
    git -C "$WORKDIR" fetch --depth 1 origin "$BRANCH"
    git -C "$WORKDIR" checkout "$BRANCH"
    git -C "$WORKDIR" reset --hard "origin/$BRANCH"
  else
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
  fi
  cd "$WORKDIR"
  DID_CLONE_OR_FETCH=1
fi

# raw.githubusercontent.com caches /raw/ for ~5 min. If the user curl-pipe-bashed
# this script, our in-memory copy may lag behind what we just cloned. Re-exec
# from disk on the first run to get the freshest logic.
if [ "$DID_CLONE_OR_FETCH" = "1" ] && [ -z "${LRD_REEXEC:-}" ] && [ -f "scripts/launch.sh" ]; then
  c_yellow "re-exec'ing from cloned scripts/launch.sh (in case raw CDN is stale)"
  export LRD_REEXEC=1
  exec bash scripts/launch.sh
fi

# 1. uv installer (idempotent).
if ! command -v uv >/dev/null 2>&1; then
  step "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { c_red "uv install failed (not on PATH)"; exit 1; }

# 2. Choose interpreter + install deps.
#    Strategy: if the system python (e.g. Colab) already has torch +
#    transformers, reuse it — avoids re-downloading ~2GB of CUDA wheels.
#    Otherwise create a fresh .venv with uv.
SYSTEM_PY="$(command -v python3 || command -v python || true)"
if [ -n "$SYSTEM_PY" ] && "$SYSTEM_PY" -c "import torch, transformers" 2>/dev/null; then
  step "using existing system Python ($SYSTEM_PY) — torch + transformers detected"
  PY="$SYSTEM_PY"
  UV_TARGET=(--system)
else
  step "creating fresh venv (.venv, python 3.11)"
  # Colab and some images set UV_SYSTEM_PYTHON=1 globally so `uv pip` ignores
  # the active venv. Clear it for this script's invocation.
  unset UV_SYSTEM_PYTHON UV_PROJECT_ENVIRONMENT
  uv venv --python 3.11 .venv
  PY="$PWD/.venv/bin/python"
  UV_TARGET=(--python "$PY")
  export VIRTUAL_ENV="$PWD/.venv"
  export PATH="$PWD/.venv/bin:$PATH"
fi

step "installing dependencies (this can take a few minutes)"
# sentencepiece/tiktoken/protobuf are required by Qwen tokenizers; spelled out
# here so the install works even if pyproject.toml on disk predates that change.
uv pip install "${UV_TARGET[@]}" \
  -e ".[hf]" \
  "trl>=0.11" "rich>=13" \
  "sentencepiece>=0.2" "tiktoken>=0.7" "protobuf>=4.21"

# Verify torch made it into the chosen interpreter.
if ! "$PY" -c "import torch" 2>/dev/null; then
  c_yellow "torch not in target interpreter after install — installing explicitly"
  uv pip install "${UV_TARGET[@]}" torch
fi

"$PY" -c "import torch; print(f'  torch={torch.__version__} cuda={torch.cuda.is_available()}')"

# 3. Sanity: GPU present?
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  c_red "no CUDA device visible to torch — training will fail"
  c_red "(set LRD_DRY_RUN=1 to set up data and skip training)"
  if [ "${LRD_DRY_RUN:-}" != "1" ]; then exit 1; fi
fi

# 4. Fetch philosophy corpus (idempotent).
mkdir -p data/philosophy logs runs
step "fetching Kant + Nietzsche from Project Gutenberg"
"$PY" scripts/fetch_philosophy.py --out data/philosophy

# 5. Build the SFT corpus (philosophy-only -> --mix 1.0, no CoT).
step "preparing SFT corpus (philosophy-only, --mix 1.0)"
"$PY" scripts/prepare_uncensored_data.py \
  --philosophy data/philosophy \
  --output data/uncensored_sft.jsonl \
  --mix 1.0

ROW_COUNT=$(wc -l < data/uncensored_sft.jsonl | tr -d ' ')
c_green "  corpus rows: $ROW_COUNT"

if [ "${LRD_DRY_RUN:-}" = "1" ]; then
  c_yellow "LRD_DRY_RUN=1, stopping before training"
  exit 0
fi

# 6. Launch trainer in background, capture stdout + structured metrics.
step "starting trainer (logs/train.stdout, logs/metrics.jsonl)"
: > logs/train.stdout
: > logs/metrics.jsonl

"$PY" -u scripts/finetune_uncensored.py \
  --config configs/finetune_uncensored.yaml \
  --metrics-jsonl logs/metrics.jsonl \
  >> logs/train.stdout 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > logs/train.pid

# Kill trainer if launch.sh exits abnormally (Ctrl-C, error).
# Normal completion (trainer exits cleanly) wait below already collected it,
# so the trap becomes a no-op.
cleanup() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    c_yellow "stopping trainer (pid=$TRAIN_PID)"
    kill -TERM "$TRAIN_PID" 2>/dev/null || true
    # Give it a chance to checkpoint
    for _ in 1 2 3 4 5; do
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$TRAIN_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

# 7. TUI (foreground). Exits when trainer exits.
#    Ctrl-C kills both the dashboard and the trainer (single intuitive abort).
#    If you want the trainer to keep running while you detach, start launch.sh
#    under tmux/nohup and disown.
if [ "${LRD_NO_TUI:-}" = "1" ]; then
  c_yellow "LRD_NO_TUI=1, tailing stdout instead of running TUI"
  tail -F logs/train.stdout &
  TAIL_PID=$!
  wait "$TRAIN_PID"; EXIT=$?
  kill "$TAIL_PID" 2>/dev/null || true
else
  step "launching TUI (Ctrl-C aborts training; run under tmux to detach safely)"
  "$PY" scripts/tui.py \
    --metrics logs/metrics.jsonl \
    --stdout logs/train.stdout \
    --pid "$TRAIN_PID" \
    --stage train || true
  # TUI exited (either trainer died, or user hit Ctrl-C). Reap the trainer.
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    c_green "TUI exited while trainer still running — waiting for it to finish..."
    wait "$TRAIN_PID"; EXIT=$?
  else
    wait "$TRAIN_PID" 2>/dev/null; EXIT=$?
  fi
fi
trap - INT TERM
if [ "$EXIT" -eq 0 ]; then
  c_green "==> done. adapter saved to runs/finetune_uncensored/"
else
  c_red "trainer exited with code $EXIT — see logs/train.stdout"
fi
exit "$EXIT"
