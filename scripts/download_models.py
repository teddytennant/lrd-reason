"""Idempotent model download. Run once on the H200 box.

Pulls:
  - Qwen/Qwen3.5-35B-A3B-FP8           (~35 GB)
  - BAAI/bge-large-en-v1.5             (~1.3 GB)
  - robbyant/lingbot-world-base-cam    (optional, kept pluggable)

Skips any model whose target dir already exists.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "BAAI/bge-large-en-v1.5",
    "robbyant/lingbot-world-base-cam",
]


def download(model_id: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. `pip install '.[hf]'`.", file=sys.stderr)
        sys.exit(1)
    target = cache_dir / model_id.replace("/", "__")
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {model_id} already present at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[pull] {model_id} -> {target}")
    snapshot_download(repo_id=model_id, local_dir=str(target), local_dir_use_symlinks=False)
    return target


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default=".model_cache")
    ap.add_argument("--only", nargs="*", default=None, help="subset of model IDs to download")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    models = args.only or DEFAULT_MODELS

    for m in models:
        path = download(m, cache)
        size_gb = _du(path) / 1e9
        print(f"   {m}: {size_gb:.2f} GB")

    total_gb = _du(cache) / 1e9
    free_gb = shutil.disk_usage(cache).free / 1e9
    print(f"\nTotal cache: {total_gb:.2f} GB. Free disk remaining: {free_gb:.2f} GB.")


if __name__ == "__main__":
    main()
