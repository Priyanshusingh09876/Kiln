"""Sequence packing for variable-length RL batches.

RL responses have a long-tailed length distribution (one 4k-token proof next
to a dozen 200-token answers), so right-padded batches burn most of their
FLOPs on pad tokens. We pack sequences into fixed-capacity bins with
first-fit-decreasing and feed the model one flattened row per bin with
``position_ids`` restarting at 0 for every sequence and a block-diagonal
attention mask (or ``cu_seqlens`` for FlashAttention's varlen entry point).

:func:`pack` / :func:`unpack` are exact inverses on the per-token tensors so
the trainer can compute log-probs on packed rows and scatter them back into
the ``[B, T-1]`` layout every loss function expects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # [P, C]
    position_ids: torch.Tensor  # [P, C]
    attention_mask: torch.Tensor  # [P, C]  (1 = real token)
    seq_index: torch.Tensor  # [P, C]  which original sequence each token belongs to, -1 = pad
    cu_seqlens: list[torch.Tensor]  # per bin, cumulative lengths (for flash varlen)
    bins: list[list[int]]  # bin -> list of original sequence indices
    lengths: torch.Tensor  # [B] original lengths

    def block_diag_mask(self) -> torch.Tensor:
        """``[P, 1, C, C]`` boolean mask: causal within each packed sequence only."""
        P, C = self.input_ids.shape
        same = self.seq_index.unsqueeze(-1) == self.seq_index.unsqueeze(-2)
        causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=self.input_ids.device))
        real = self.seq_index.unsqueeze(-1) >= 0
        m = same & causal & real
        diag = torch.eye(C, dtype=torch.bool, device=self.input_ids.device)
        return (m | diag).unsqueeze(1)


def first_fit_decreasing(lengths: list[int], capacity: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins: list[list[int]] = []
    remaining: list[int] = []
    for i in order:
        L = lengths[i]
        if L > capacity:
            raise ValueError(f"sequence {i} of length {L} exceeds capacity {capacity}")
        for b, rem in enumerate(remaining):
            if rem >= L:
                bins[b].append(i)
                remaining[b] -= L
                break
        else:
            bins.append([i])
            remaining.append(capacity - L)
    return bins


def pack(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, capacity: int, pad_id: int = 0
) -> PackedBatch:
    B, T = input_ids.shape
    lengths = attention_mask.sum(-1).tolist()
    bins = first_fit_decreasing(lengths, capacity)
    P = len(bins)
    dev = input_ids.device
    out_ids = torch.full((P, capacity), pad_id, dtype=input_ids.dtype, device=dev)
    pos = torch.zeros((P, capacity), dtype=torch.long, device=dev)
    am = torch.zeros((P, capacity), dtype=torch.long, device=dev)
    sidx = torch.full((P, capacity), -1, dtype=torch.long, device=dev)
    cu = []
    for b, members in enumerate(bins):
        cursor = 0
        offsets = [0]
        for i in members:
            L = int(lengths[i])
            out_ids[b, cursor : cursor + L] = input_ids[i, :L]
            pos[b, cursor : cursor + L] = torch.arange(L, device=dev)
            am[b, cursor : cursor + L] = 1
            sidx[b, cursor : cursor + L] = i
            cursor += L
            offsets.append(cursor)
        cu.append(torch.tensor(offsets, dtype=torch.int32, device=dev))
    return PackedBatch(out_ids, pos, am, sidx, cu, bins, torch.tensor(lengths, dtype=torch.long, device=dev))


def unpack(packed: PackedBatch, per_token: torch.Tensor, T: int, shift: bool = True) -> torch.Tensor:
    """Scatter a ``[P, C]`` (or ``[P, C-1]`` when ``shift``) tensor back to ``[B, T]`` / ``[B, T-1]``.

    With ``shift=True`` the input is aligned to *predicted* tokens, i.e. index
    ``c`` corresponds to packed position ``c+1``; we therefore map each packed
    position ``c+1`` back to ``(seq, pos)`` and store at ``pos-1``. Tokens at
    position 0 of each sequence (the first prompt token) have no prediction
    and are skipped, exactly like the ``[B, T-1]`` convention.
    """
    B = int(packed.lengths.shape[0])
    width = T - 1 if shift else T
    out = torch.zeros((B, width), dtype=per_token.dtype, device=per_token.device)
    if shift:
        seq = packed.seq_index[:, 1:]
        pos = packed.position_ids[:, 1:]
        valid = (seq >= 0) & (pos > 0)
        out[seq[valid], pos[valid] - 1] = per_token[valid]
    else:
        seq = packed.seq_index
        pos = packed.position_ids
        valid = seq >= 0
        out[seq[valid], pos[valid]] = per_token[valid]
    return out


def packing_efficiency(packed: PackedBatch) -> float:
    return float(packed.attention_mask.sum().item()) / float(packed.attention_mask.numel())
