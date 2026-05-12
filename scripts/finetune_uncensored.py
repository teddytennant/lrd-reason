"""LoRA SFT on HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive.

A separate, simpler track from the main lrd-reason pipeline: no recurrent state,
no diffusion module, just QLoRA/LoRA fine-tuning of the 4B uncensored base on
the same data philosophy (Kant + Nietzsche raw text for voice, COT_SYSTEM_PROMPT
traces for reasoning). The intent is a small, deployable model that already
embodies the constitution at the surface level without the depth of the 35B +
latent-reasoning system.

Why the "Aggressive" uncensored variant: principle VII (Truth Over Obedience)
asks the model to correct the human rather than soften or hedge. Fine-tuning a
base that hasn't been RLHF-trained into reflexive deflection is the cheap path;
on a vanilla instruct base the SFT loss would have to overcome the safety prior.

Inputs:
  --train JSONL    Output of scripts/prepare_uncensored_data.py. Each line is
                   either {"kind": "raw", "text": ...} or
                   {"kind": "chat", "messages": [...]}.
  --config YAML    See configs/finetune_uncensored.yaml.
  --output DIR     LoRA adapter checkpoints land here.

This script is NOT executed by pytest; it requires CUDA + ~16GB VRAM (bf16 LoRA)
or ~10GB (4-bit QLoRA via bitsandbytes). The data-formatting helpers it imports
from prepare_uncensored_data.py ARE tested on CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FineTuneSpec:
    model_id: str = "HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive"
    trust_remote_code: bool = True

    # Training
    output_dir: str = "runs/finetune_uncensored"
    seed: int = 1337
    max_seq_len: int = 2048
    per_device_batch_size: int = 1
    grad_accum_steps: int = 16
    num_train_epochs: float = 2.0
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    grad_checkpointing: bool = True
    log_every: int = 25
    save_every: int = 500
    bf16: bool = True
    quantize_4bit: bool = False  # set true for QLoRA on <16GB GPUs

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Data
    train_jsonl: str = "data/uncensored_sft.jsonl"
    eval_jsonl: str | None = None
    pack_raw_with_eos: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> FineTuneSpec:
        with open(path) as f:
            blob = yaml.safe_load(f) or {}
        return cls(**blob)


def format_row(row: dict, tokenizer: Any) -> str:
    """Render one corpus row into the string fed to SFTTrainer.

    - kind=chat → apply the model's chat template (Qwen3 chat template).
    - kind=raw  → emit the text verbatim plus EOS so packing terminates cleanly.

    Pulled out for unit-testing without loading a real tokenizer (caller passes
    a duck-typed object with .apply_chat_template and .eos_token).
    """
    kind = row.get("kind", "chat")
    if kind == "chat":
        return tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    if kind == "raw":
        eos = getattr(tokenizer, "eos_token", "") or ""
        return row["text"].rstrip() + ("\n" + eos if eos else "\n")
    raise ValueError(f"unknown row kind: {kind!r}")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_lora_model(spec: FineTuneSpec):
    """Heavy imports are inside this function so unit tests can import the module
    without transformers/peft/bitsandbytes installed."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "torch_dtype": torch.bfloat16 if spec.bf16 else torch.float16,
    }
    if spec.quantize_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)
    if spec.quantize_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=spec.grad_checkpointing,
        )
    elif spec.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=spec.lora_r,
        lora_alpha=spec.lora_alpha,
        lora_dropout=spec.lora_dropout,
        bias="none",
        target_modules=spec.lora_targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


class JsonlMetricsCallback:
    """Append trainer log dicts to a JSONL file so external watchers (e.g. tui.py)
    can read training progress without parsing stdout.

    Implemented as duck-typed TrainerCallback to keep this module importable
    without transformers installed (the heavy imports stay inside train()).
    """

    def __init__(self, path: str | Path) -> None:
        from pathlib import Path as _P
        self.path = _P(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        if logs is None:
            return
        import json as _json
        import time as _time
        record = {
            "step": getattr(state, "global_step", None),
            "max_steps": getattr(state, "max_steps", None),
            "epoch": getattr(state, "epoch", None),
            "wall": _time.time(),
            **logs,
        }
        with self.path.open("a") as f:
            f.write(_json.dumps(record) + "\n")

    # Other callback hooks are no-ops; HF TrainerCallback tolerates missing methods.
    def on_train_begin(self, args, state, control, **kwargs): return None  # noqa: E704
    def on_train_end(self, args, state, control, **kwargs): return None  # noqa: E704
    def on_step_end(self, args, state, control, **kwargs): return None  # noqa: E704


def train(spec: FineTuneSpec, train_path: Path, eval_path: Path | None,
          metrics_jsonl: Path | None = None) -> None:
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"missing dep: {e}. install with `uv pip install '.[hf]' trl bitsandbytes`",
              file=sys.stderr)
        sys.exit(1)

    model, tokenizer = build_lora_model(spec)

    raw_train = load_jsonl(train_path)
    train_texts = [{"text": format_row(r, tokenizer)} for r in raw_train]
    train_ds = Dataset.from_list(train_texts)

    eval_ds = None
    if eval_path is not None:
        raw_eval = load_jsonl(eval_path)
        eval_texts = [{"text": format_row(r, tokenizer)} for r in raw_eval]
        eval_ds = Dataset.from_list(eval_texts)

    sft_cfg = SFTConfig(
        output_dir=spec.output_dir,
        seed=spec.seed,
        num_train_epochs=spec.num_train_epochs,
        per_device_train_batch_size=spec.per_device_batch_size,
        gradient_accumulation_steps=spec.grad_accum_steps,
        learning_rate=spec.lr,
        weight_decay=spec.weight_decay,
        warmup_ratio=spec.warmup_ratio,
        lr_scheduler_type=spec.lr_scheduler,
        logging_steps=spec.log_every,
        save_steps=spec.save_every,
        save_total_limit=3,
        bf16=spec.bf16,
        max_seq_length=spec.max_seq_len,
        packing=True,
        dataset_text_field="text",
        gradient_checkpointing=spec.grad_checkpointing,
        report_to=[],
    )

    callbacks = []
    if metrics_jsonl is not None:
        callbacks.append(JsonlMetricsCallback(metrics_jsonl))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks or None,
    )
    trainer.train()
    trainer.save_model(spec.output_dir)
    tokenizer.save_pretrained(spec.output_dir)
    print(f"[done] adapter + tokenizer saved to {spec.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--train", type=Path, default=None,
                    help="override spec.train_jsonl")
    ap.add_argument("--eval", type=Path, default=None,
                    help="override spec.eval_jsonl")
    ap.add_argument("--output", type=Path, default=None,
                    help="override spec.output_dir")
    ap.add_argument("--metrics-jsonl", type=Path, default=None,
                    help="append per-log-step training metrics to this JSONL (for tui.py)")
    args = ap.parse_args()

    spec = FineTuneSpec.from_yaml(args.config)
    if args.train is not None:
        spec.train_jsonl = str(args.train)
    if args.eval is not None:
        spec.eval_jsonl = str(args.eval)
    if args.output is not None:
        spec.output_dir = str(args.output)

    train_path = Path(spec.train_jsonl)
    eval_path = Path(spec.eval_jsonl) if spec.eval_jsonl else None
    if not train_path.exists():
        raise SystemExit(f"train jsonl not found: {train_path}")
    train(spec, train_path, eval_path, metrics_jsonl=args.metrics_jsonl)


if __name__ == "__main__":
    main()
