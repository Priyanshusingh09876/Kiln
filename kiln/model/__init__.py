"""Model loading: the built-in tiny transformer or any HF causal LM."""

from __future__ import annotations

import torch
import torch.nn as nn

from kiln.config.schema import ModelConfig
from kiln.model.tiny import TinyConfig, TinyLM, ValueHead

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str, device: torch.device | str | None = None) -> torch.dtype:
    dt = _DTYPES[name]
    # bf16 matmuls on CPU are slow and some ops are unsupported; keep CPU runs in fp32.
    if (device is None and not torch.cuda.is_available()) or str(device) == "cpu":
        return torch.float32
    return dt


def tiny_config_from(cfg: ModelConfig) -> TinyConfig:
    return TinyConfig(
        vocab_size=cfg.tiny_vocab,
        hidden=cfg.tiny_hidden,
        layers=cfg.tiny_layers,
        heads=cfg.tiny_heads,
        max_seq=cfg.tiny_max_seq,
    )


def load_policy(cfg: ModelConfig, device: torch.device | str = "cpu") -> nn.Module:
    """Return a causal LM whose ``forward(input_ids, attention_mask, position_ids)`` yields logits."""
    dtype = resolve_dtype(cfg.dtype, device)
    if cfg.name_or_path == "tiny":
        model = TinyLM(tiny_config_from(cfg)).to(device=device, dtype=dtype)
        return model
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'kiln-rl[train]' to load HuggingFace models") from e
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name_or_path,
        torch_dtype=dtype,
        attn_implementation={"sdpa": "sdpa", "flash": "flash_attention_2", "eager": "eager"}[cfg.attn_impl],
        trust_remote_code=cfg.trust_remote_code,
    )
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return HFWrapper(model).to(device)


class HFWrapper(nn.Module):
    """Normalises HF's ``CausalLMOutput`` to a bare logits tensor and exposes ``forward_hidden``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kw):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kw
        ).logits

    def forward_hidden(self, input_ids, attention_mask=None, position_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)


def hidden_size_of(model: nn.Module) -> int:
    if isinstance(model, TinyLM):
        return model.cfg.hidden
    if isinstance(model, HFWrapper):
        return model.hidden_size
    raise TypeError(f"cannot infer hidden size of {type(model).__name__}")


__all__ = ["HFWrapper", "TinyConfig", "TinyLM", "ValueHead", "hidden_size_of", "load_policy", "resolve_dtype"]
