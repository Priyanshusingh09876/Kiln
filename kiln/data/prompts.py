"""Prompt datasets for online RL and preference pairs for DPO.

Input is JSONL (one object per line) or Parquet. For RL each row needs a
prompt (string, or a chat ``messages`` list) and whatever the reward function
needs (typically ``answer``). Tokenisation is done once up front and the
dataset is iterated by an infinite, seeded, epoch-aware sampler so the
rollout workers can keep pulling prompts for as long as the trainer runs.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kiln.config.schema import DataConfig


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def eos_id(self) -> int: ...


class ByteTokenizer:
    """Deterministic byte-level tokenizer (vocab 256 + specials) for tests / the tiny model."""

    PAD, EOS, BOS = 256, 257, 258

    def __init__(self, vocab_size: int = 512):
        assert vocab_size >= 259
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [self.BOS] + list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    @property
    def pad_id(self) -> int:
        return self.PAD

    @property
    def eos_id(self) -> int:
        return self.EOS


class HFTokenizerAdapter:
    def __init__(self, name_or_path: str, chat_template: bool = True):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.chat_template = chat_template and self.tok.chat_template is not None

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=True)

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        if self.chat_template:
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return self.encode("\n".join(m["content"] for m in messages))

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    @property
    def pad_id(self) -> int:
        return int(self.tok.pad_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.tok.eos_token_id)


@dataclass
class Prompt:
    uid: int
    text: str
    ids: list[int]
    answer: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".jsonl":
        with p.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(p).to_pylist()
    if p.suffix == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"unsupported dataset format {p.suffix}")


class PromptDataset:
    def __init__(self, cfg: DataConfig, tokenizer: Tokenizer, rows: list[dict[str, Any]] | None = None):
        self.cfg = cfg
        self.tok = tokenizer
        rows = rows if rows is not None else _read_rows(cfg.train_path)
        self.prompts: list[Prompt] = []
        for i, r in enumerate(rows):
            raw = r[cfg.prompt_key]
            if isinstance(raw, list):  # chat messages
                text = "\n".join(m.get("content", "") for m in raw)
                ids = (
                    tokenizer.encode_messages(raw)
                    if hasattr(tokenizer, "encode_messages")
                    else tokenizer.encode(text)
                )
            else:
                text = str(raw)
                ids = tokenizer.encode(text)
            ids = ids[-cfg.max_prompt_len :]
            answer = r.get(cfg.answer_key)
            meta = {k: v for k, v in r.items() if k not in (cfg.prompt_key, cfg.answer_key)}
            self.prompts.append(Prompt(i, text, ids, None if answer is None else str(answer), meta))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> Prompt:
        return self.prompts[i]

    def sampler(self, seed: int, rank: int = 0, world_size: int = 1) -> Iterator[Prompt]:
        """Infinite epoch-based sampler, sharded by rank, reproducible from ``seed``."""
        rng = random.Random(seed)
        epoch = 0
        while True:
            order = list(range(len(self.prompts)))
            if self.cfg.shuffle:
                rng.shuffle(order)
            for j in order[rank::world_size]:
                yield self.prompts[j]
            epoch += 1

    def batches(
        self, batch_size: int, seed: int, rank: int = 0, world_size: int = 1
    ) -> Iterator[list[Prompt]]:
        it = self.sampler(seed, rank, world_size)
        while True:
            yield [next(it) for _ in range(batch_size)]


@dataclass
class PreferencePair:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


class PreferenceDataset:
    """For DPO: rows with ``prompt``, ``chosen``, ``rejected`` text fields."""

    def __init__(self, path: str | Path, tokenizer: Tokenizer, max_len: int = 4096):
        self.pairs: list[PreferencePair] = []
        for r in _read_rows(path):
            p = tokenizer.encode(r["prompt"])
            c = tokenizer.encode(r["chosen"])[1:] + [tokenizer.eos_id]
            j = tokenizer.encode(r["rejected"])[1:] + [tokenizer.eos_id]
            self.pairs.append(PreferencePair(p[-max_len:], c[: max_len - len(p)], j[: max_len - len(p)]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> PreferencePair:
        return self.pairs[i]
