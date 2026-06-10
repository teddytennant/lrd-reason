"""Encode gold CoT traces to target latents for Stage 1/2 training.

After `generate_data.py` produces cot.*.jsonl, this script runs the configured
encoder (BGE/Stub/...) + EncoderProjector over the gold_cot fields and saves
a dict {prompt_sha1: latent_tensor} as .pt. The result is passed as
data.latents_path so LatentPairDataset can supply exact targets instead of
the hash-seeded random fallback.

Run on the same machine as training (or any with the encoder deps). CPU works
for stub; BGE benefits from CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lrd_reason.config import load_config
from lrd_reason.data.dataset import _prompt_hash
from lrd_reason.models.encoder import EncoderProjector, build_encoder


def _encode_batch(
    encoder, projector, texts: list[str], device: torch.device
) -> torch.Tensor:
    raw = encoder.encode(texts).to(device)
    return projector(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--inputs", nargs="+", required=True, help="one or more cot.*.jsonl")
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: from config)")
    args = ap.parse_args()

    spec = load_config(args.config)
    device_str = args.device or spec.run.device
    device = torch.device(device_str)

    enc = build_encoder(spec.model.encoder, device=device_str)
    proj = EncoderProjector(
        raw_dim=spec.model.encoder.raw_dim,
        latent_dim=spec.model.latent_dim,
        projector_path=spec.model.encoder.projector_path,
    ).to(device)

    latents: dict[str, torch.Tensor] = {}
    total = 0
    for inp in args.inputs:
        p = Path(inp)
        if not p.exists():
            print(f"[warn] missing {p}, skipping")
            continue
        rows = [json.loads(line) for line in p.open()]
        prompts = []
        texts = []
        for r in rows:
            pr = r["prompt"]
            cot = (r.get("gold_cot") or "").strip()
            text = cot if cot else pr  # prefer gold CoT; fall back to prompt
            h = _prompt_hash(pr)
            if h in latents:
                continue
            prompts.append(pr)
            texts.append(text)
            if len(texts) >= args.batch_size:
                with torch.no_grad():
                    vecs = _encode_batch(enc, proj, texts, device)
                for pr_i, v in zip(prompts, vecs, strict=True):
                    latents[_prompt_hash(pr_i)] = v.cpu()
                total += len(texts)
                prompts, texts = [], []
        if texts:
            with torch.no_grad():
                vecs = _encode_batch(enc, proj, texts, device)
            for pr_i, v in zip(prompts, vecs, strict=True):
                latents[_prompt_hash(pr_i)] = v.cpu()
            total += len(texts)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents, out_p)
    print(f"[done] {len(latents)} unique targets ({total} rows processed) -> {out_p}")


if __name__ == "__main__":
    main()
