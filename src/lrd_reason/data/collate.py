"""Custom collate function for diffusion-aware batching.

Batches a list of LatentPairDataset items into stacked tensors. Multi-turn batching
is the caller's responsibility (group examples by session_id before sending here);
this collator just stacks whatever it's given.
"""

from __future__ import annotations

from typing import Any

import torch


def collate_pairs(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": [it["prompt"] for it in items],
        "gold_cots": [it["gold_cot"] for it in items],
        "gold_answers": [it["gold_answer"] for it in items],
        "session_ids": [it["session_id"] for it in items],
        "turn_idxs": torch.tensor([it["turn_idx"] for it in items], dtype=torch.long),
        "target_latents": torch.stack([it["target_latent"] for it in items], dim=0),
        "task_embeds": torch.stack([it["task_embed"] for it in items], dim=0),
    }
