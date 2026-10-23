"""Learned reward model scorer (sequence classification head)."""

from __future__ import annotations

import torch


class RewardModelScorer:
    def __init__(
        self, name_or_path: str, batch_size: int = 64, device: str | None = None, max_len: int = 4096
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name_or_path)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                name_or_path,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                num_labels=1,
            )
            .to(self.device)
            .eval()
        )
        self.batch_size = batch_size
        self.max_len = max_len

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for s in range(0, len(pairs), self.batch_size):
            chunk = pairs[s : s + self.batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}, {"role": "assistant", "content": r}], tokenize=False
                )
                if self.tok.chat_template
                else f"{p}\n{r}"
                for p, r in chunk
            ]
            enc = self.tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1).float()
            out.extend(logits.tolist())
        return out
