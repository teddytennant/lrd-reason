"""Inference-time wrapper around LRDPipeline.

Loads checkpoints (stage1 + optional stage2), exposes `chat(session_id, prompt, K)`
that maintains persistent recurrent state across turns via the state_store.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..config import RunSpec, load_config
from ..models.pipeline import LRDPipeline
from .state_store import load_state, save_state


class InferenceRunner:
    def __init__(
        self,
        spec: RunSpec,
        stage1_ckpt: str | Path | None = None,
        stage2_ckpt: str | Path | None = None,
        sessions_dir: str | Path = "sessions",
    ) -> None:
        self.spec = spec
        self.device = torch.device(spec.run.device)
        self.dtype = getattr(torch, spec.run.dtype)
        self.sessions_dir = Path(sessions_dir)
        self.pipeline = LRDPipeline(
            model_cfg=spec.model,
            ablation=spec.ablation,
            task_dim=spec.data.task_embed_dim,
            device=spec.run.device,
            dtype=self.dtype,
        ).to(self.device)
        self.pipeline.eval()

        if stage1_ckpt is not None:
            self._load(stage1_ckpt)
        if stage2_ckpt is not None:
            self._load(stage2_ckpt)

    def _load(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            link = path / "latest.symlink"
            if not link.exists():
                raise FileNotFoundError(f"no latest.symlink in {path}")
            path = (path / path.readlink() if path.is_symlink() else (link.parent / link.readlink()))
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.pipeline.load_state_dict(payload["model"], strict=False)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        llm = self.pipeline.llm
        if hasattr(llm, "tokenize"):
            return llm.tokenize([prompt]).to(self.device)  # type: ignore[attr-defined]
        # Stub LLM: deterministic toy tokenization.
        max_len = 8
        vocab_size = getattr(llm, "vocab_size", 64)
        ids = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        for j, ch in enumerate(prompt[:max_len]):
            ids[0, j] = ord(ch) % vocab_size
        return ids

    @torch.no_grad()
    def chat(
        self,
        session_id: str,
        prompt: str,
        num_diffusion_steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        prev = load_state(self.sessions_dir, session_id)
        prev_state = prev.state.to(self.device).unsqueeze(0) if prev is not None else None
        turn_idx = (prev.turn_idx + 1) if prev is not None else 0

        input_ids = self._tokenize_prompt(prompt)
        out_ids, new_state, plan = self.pipeline.generate(
            prompts=[prompt],
            input_ids=input_ids,
            prev_state=prev_state,
            task_embed=None,
            num_diffusion_steps=num_diffusion_steps,
            max_new_tokens=max_new_tokens,
        )

        save_state(
            self.sessions_dir,
            session_id,
            new_state.squeeze(0),
            turn_idx=turn_idx,
            meta={"last_prompt": prompt},
        )

        llm = self.pipeline.llm
        if hasattr(llm, "decode"):
            return llm.decode(out_ids)[0]  # type: ignore[attr-defined]
        # Stub LLM: turn ids into a string of characters.
        ids = out_ids[0].tolist()
        return "".join(chr((i % 95) + 32) for i in ids)


def runner_from_config(path: str | Path, **kwargs) -> InferenceRunner:
    spec = load_config(path)
    return InferenceRunner(spec=spec, **kwargs)
