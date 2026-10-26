"""In-process rollout engine backed by :class:`kiln.model.TinyLM`.

It holds its *own copy* of the policy so weight sync is a real operation
(``update_weights`` copies tensors in), and it can be told to lag behind the
trainer, which is how the staleness / importance-weighting tests inject
controlled off-policy-ness. Generation is the naive sampling loop; this is a
correctness oracle, not a throughput engine.
"""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Iterable

import torch

from kiln.config.schema import SamplingConfig
from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.types import Sample


class MockEngine(RolloutEngine):
    name = "mock"

    def __init__(
        self,
        model: torch.nn.Module,
        eos_id: int | None = None,
        seed: int = 0,
        device: str = "cpu",
        latency_ms_per_token: float = 0.0,
        median_tokens: int = 600,
        length_sigma: float = 1.0,
    ):
        super().__init__()
        self.model = copy.deepcopy(model).to(device).eval()
        self.eos_id = eos_id
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self._tokens_generated = 0
        self._gen_seconds = 0.0
        self.latency_ms_per_token = latency_ms_per_token
        self.median_tokens = median_tokens
        self.length_sigma = length_sigma
        self._lat_rng = random.Random(seed + 7919)
        self.simulated_seconds = 0.0

    @torch.no_grad()
    def generate(self, requests: list[GenerationRequest], sampling: SamplingConfig) -> list[Sample]:
        t0 = time.perf_counter()
        out: list[Sample] = []
        version = self.policy_version
        for req in requests:
            n = req.n or sampling.n
            prompt = torch.tensor([req.prompt_ids] * n, dtype=torch.long, device=self.device)
            resp, lp = self.model.generate(
                prompt,
                max_new_tokens=sampling.max_new_tokens,
                temperature=sampling.temperature,
                top_k=sampling.top_k,
                eos_id=self.eos_id,
                generator=self.gen,
            )
            for i in range(n):
                ids = resp[i].tolist()
                lps = lp[i].tolist()
                if self.eos_id is not None and self.eos_id in ids:
                    cut = ids.index(self.eos_id) + 1
                    ids, lps = ids[:cut], lps[:cut]
                    finish = "stop"
                else:
                    finish = "length"
                self._tokens_generated += len(ids)
                out.append(
                    Sample(
                        prompt_ids=list(req.prompt_ids),
                        response_ids=ids,
                        behavior_logprobs=lps,
                        group_id=req.group_id,
                        policy_version=version,
                        prompt_text=req.prompt_text,
                        finish_reason=finish,
                        meta=dict(req.meta or {}),
                    )
                )
        if self.latency_ms_per_token > 0:
            # a generate() call returns when its longest response finishes; sleeping releases the GIL,
            # so the trainer thread really runs concurrently, as it would with the engine on other GPUs
            n_resp = len(out)
            lengths = [
                self.median_tokens * math.exp(self.length_sigma * self._lat_rng.gauss(0, 1))
                for _ in range(n_resp)
            ]
            delay = max(lengths) * self.latency_ms_per_token / 1e3
            self.simulated_seconds += delay
            time.sleep(delay)
        self._gen_seconds += time.perf_counter() - t0
        return out

    @torch.no_grad()
    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        sd = self.model.state_dict()
        for name, t in named_tensors:
            if name in sd:
                sd[name].copy_(t.to(sd[name].device, sd[name].dtype))
        self.policy_version = version

    def state_dict(self) -> dict:
        return {"rng": self.gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        if "rng" in state:
            # checkpoints are loaded onto the trainer device (cuda); a CPU generator
            # only accepts a CPU ByteTensor
            self.gen.set_state(torch.as_tensor(state["rng"]).to("cpu", torch.uint8))

    def stats(self) -> dict[str, float]:
        return {
            "rollout/tokens": float(self._tokens_generated),
            "rollout/tok_per_s": self._tokens_generated / max(self._gen_seconds, 1e-9),
        }
