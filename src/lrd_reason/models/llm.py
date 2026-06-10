"""LLM wrappers.

Two implementations:

- StubLLM: random-init nn.Embedding + nn.Linear "vocab head" + identity body. Lets
  the full pipeline forward end-to-end on CPU for smoke tests. Loss is meaningful
  (cross-entropy on random targets is non-zero), so optimizer updates produce a
  decreasing loss curve.

- HFLLM: lazy-loads a Hugging Face causal LM. Frozen on construction; LoRA is
  attached externally via adapter.attach_lora. Exposes hidden_size for adapter
  configuration.

Both expose:
    embed(input_ids) -> [B, T, H]
    forward_with_prefix(prefix_embeds, prompt_ids, labels=None) -> {logits, loss}
    generate(prompt_ids, prefix_embeds, max_new_tokens) -> ids
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import LLMConfig


@dataclass
class LLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class StubLLM(nn.Module):
    """Trainable toy LM for CPU smoke tests."""

    def __init__(self, vocab_size: int, hidden_size: int, max_new_tokens: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_new_tokens = max_new_tokens
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.body = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> LLMOutput:
        tok_embeds = self.embed_tokens(input_ids)
        h = torch.cat([prefix_embeds, tok_embeds], dim=1)
        h = self.body(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            # Shift labels to align with logit positions over the input_ids portion.
            n_prefix = prefix_embeds.shape[1]
            shift_logits = logits[:, n_prefix - 1 : -1].contiguous()
            shift_labels = labels.contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return LLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        max_new = max_new_tokens or self.max_new_tokens
        ids = input_ids
        for _ in range(max_new):
            out = self.forward_with_prefix(prefix_embeds, ids)
            next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


class HFLLM(nn.Module):
    """Hugging Face causal LM wrapper. Lazy-loaded, frozen on construction."""

    def __init__(self, cfg: LLMConfig, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.cfg = cfg
        self.device_str = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None
        self._hidden_size: int | None = None

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            self._ensure_loaded()
        return self._hidden_size  # type: ignore[return-value]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for HFLLM. Install with `pip install '.[hf]'`."
            ) from e
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_id, trust_remote_code=self.cfg.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.hf_id,
            trust_remote_code=self.cfg.trust_remote_code,
            torch_dtype=self.dtype,
            device_map=self.device_str if self.device_str != "cpu" else None,
        )
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()
        self._hidden_size = self._model.config.hidden_size

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._ensure_loaded()
        return self._model.get_input_embeddings()(input_ids)  # type: ignore[union-attr]

    def forward_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> LLMOutput:
        self._ensure_loaded()
        tok_embeds = self.embed(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        # Labels need a leading prefix-length of -100 so we don't train on the prefix.
        if labels is not None:
            n_prefix = prefix_embeds.shape[1]
            pad = labels.new_full((labels.shape[0], n_prefix), -100)
            labels = torch.cat([pad, labels], dim=1)
        out = self._model(  # type: ignore[misc]
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return LLMOutput(logits=out.logits, loss=out.loss if labels is not None else None)

    @torch.no_grad()
    def generate(
        self,
        prefix_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        self._ensure_loaded()
        max_new = max_new_tokens or self.cfg.max_new_tokens
        tok_embeds = self.embed(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        return self._model.generate(  # type: ignore[misc]
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new,
            do_sample=False,
        )

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        self._ensure_loaded()
        return self._tokenizer(  # type: ignore[union-attr]
            texts, padding=True, truncation=True, return_tensors="pt"
        ).input_ids

    def encode_text(self, text: str) -> list[int]:
        """Unpadded, untruncated id stream — used for corpus packing in pretrain."""
        self._ensure_loaded()
        return self._tokenizer.encode(text, add_special_tokens=False)  # type: ignore[union-attr]

    def decode(self, ids: torch.Tensor) -> list[str]:
        self._ensure_loaded()
        return self._tokenizer.batch_decode(ids, skip_special_tokens=True)  # type: ignore[union-attr]


def build_llm(cfg: LLMConfig, hidden_size: int, device: str = "cpu", dtype: torch.dtype = torch.float32):
    if cfg.kind == "stub":
        return StubLLM(vocab_size=cfg.vocab_size, hidden_size=hidden_size, max_new_tokens=cfg.max_new_tokens)
    if cfg.kind == "hf":
        return HFLLM(cfg=cfg, device=device, dtype=dtype)
    raise ValueError(f"unknown llm kind: {cfg.kind}")
