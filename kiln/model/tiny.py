"""A small Llama-style decoder built on Kiln's kernels.

Two jobs:

1. Be the *reference architecture* the kernels are developed against
   (RMSNorm -> RoPE -> SDPA attention -> SwiGLU MLP, pre-norm, tied embeddings).
2. Be the model used by the CPU test-suite and the ``mock`` rollout engine so
   the whole async GRPO loop can be exercised end-to-end in a few seconds
   without a GPU or a HuggingFace download.

Production runs load a HF checkpoint through :func:`kiln.model.load_policy`;
this class is deliberately small (no KV cache, no MoE) so it stays readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kiln.kernels import RMSNorm, apply_rope, rope_tables


@dataclass
class TinyConfig:
    vocab_size: int = 512
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    max_seq: int = 256
    mlp_ratio: float = 2.67
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads


class Attention(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.hidden, 3 * cfg.hidden, bias=False)
        self.o = nn.Linear(cfg.hidden, cfg.hidden, bias=False)

    def forward(self, x, cos, sin, positions, attn_mask):
        B, T, _ = x.shape
        H, D = self.cfg.heads, self.cfg.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, D).unbind(2)
        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,T,D]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        return self.o(out.transpose(1, 2).reshape(B, T, H * D))


class MLP(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        inner = int(cfg.hidden * cfg.mlp_ratio)
        inner = (inner + 7) // 8 * 8
        self.gate = nn.Linear(cfg.hidden, inner, bias=False)
        self.up = nn.Linear(cfg.hidden, inner, bias=False)
        self.down = nn.Linear(inner, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.hidden)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.hidden)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, positions, attn_mask):
        x = x + self.attn(self.n1(x), cos, sin, positions, attn_mask)
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.hidden)
        self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        cos, sin = rope_tables(cfg.max_seq, cfg.head_dim, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        block_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
        **_: object,
    ) -> torch.Tensor:
        """Return logits ``[B, T, V]``.

        ``attention_mask`` (``[B, T]``) masks padding; ``block_mask``
        (``[B, 1, T, T]`` bool, from :meth:`PackedBatch.block_diag_mask`) is
        used instead when sequences are packed.
        """
        B, T = input_ids.shape
        x = self.embed(input_ids)
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        attn_mask = None
        if block_mask is not None:
            attn_mask = block_mask
        elif attention_mask is not None:
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            key_ok = attention_mask.bool()[:, None, None, :]  # [B,1,1,T]
            attn_mask = causal[None, None] & key_ok
            # a fully-masked row produces NaN in SDPA; let padded queries attend to themselves
            diag = torch.eye(T, dtype=torch.bool, device=x.device)[None, None]
            attn_mask = attn_mask | diag
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin, position_ids, attn_mask)
        h = self.norm(x)
        return h if return_hidden else self.lm_head(h)

    def forward_hidden(self, input_ids, attention_mask=None, position_ids=None, block_mask=None):
        """Final normalised hidden states ``[B, T, H]`` (critic input).

        Goes through ``__call__`` so FSDP2 pre/post-forward hooks still unshard
        the root parameters.
        """
        return self(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            block_mask=block_mask,
            return_hidden=True,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = -1,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive sampling loop. Returns ``(response_ids [B, L], logprobs [B, L])``."""
        ids = prompt_ids.clone()
        B = ids.shape[0]
        out_tokens, out_lp = [], []
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :].float()
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            if top_k > 0:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            logp = torch.log_softmax(logits, dim=-1)
            if not torch.isfinite(logp).any(dim=-1).all():
                raise FloatingPointError("non-finite logits in generate(); policy weights are NaN/inf")
            nxt = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
            lp = logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
            nxt = torch.where(done, torch.zeros_like(nxt), nxt)
            out_tokens.append(nxt)
            out_lp.append(lp)
            ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
            if eos_id is not None:
                done = done | (nxt == eos_id)
                if bool(done.all()):
                    break
        return torch.stack(out_tokens, 1), torch.stack(out_lp, 1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self) -> float:
        """6*N + attention term, used by the MFU calculator."""
        n = self.num_params()
        attn = 12 * self.cfg.layers * self.cfg.hidden * self.cfg.max_seq
        return 6.0 * n + attn


class ValueHead(nn.Module):
    """Scalar critic on top of any model exposing ``hidden`` states via ``forward_hidden``."""

    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.proj.weight, std=1 / math.sqrt(hidden))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1)
