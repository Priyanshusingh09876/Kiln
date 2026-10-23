"""SGLang rollout engine over HTTP (one or more ``sglang.launch_server`` instances).

Requests are fanned out across servers round-robin and issued concurrently.
Weight updates use SGLang's ``/update_weights_from_disk`` (simple, works
across nodes) and ``/pause_generation`` / ``/continue_generation`` implement
partial rollouts without the engine draining its queue.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Iterable

import torch

from kiln.config.schema import RolloutConfig, SamplingConfig
from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.types import Sample


class SGLangEngine(RolloutEngine):
    name = "sglang"

    def __init__(self, cfg: RolloutConfig, timeout_s: float = 600.0):
        super().__init__()
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'kiln-rl[sglang]'") from e
        self._httpx = httpx
        self.urls = list(cfg.sglang_urls)
        self.timeout = timeout_s
        self._rr = 0
        self._tokens = 0
        self._seconds = 0.0

    def _next_url(self) -> str:
        url = self.urls[self._rr % len(self.urls)]
        self._rr += 1
        return url

    async def _one(
        self, client, req: GenerationRequest, sampling: SamplingConfig, n: int, version: int
    ) -> list[Sample]:
        payload = {
            "input_ids": req.prompt_ids,
            "sampling_params": {
                "n": n,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "max_new_tokens": sampling.max_new_tokens,
                "stop": sampling.stop,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        r = await client.post(f"{self._next_url()}/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        items = body if isinstance(body, list) else [body]
        out = []
        for it in items:
            ids = it["output_ids"] if "output_ids" in it else it["meta_info"]["output_token_ids"]
            lps = [x[0] for x in it["meta_info"]["output_token_logprobs"]]
            self._tokens += len(ids)
            out.append(
                Sample(
                    prompt_ids=list(req.prompt_ids),
                    response_ids=list(ids),
                    behavior_logprobs=lps,
                    group_id=req.group_id,
                    policy_version=version,
                    prompt_text=req.prompt_text,
                    response_text=it.get("text", ""),
                    finish_reason=it["meta_info"].get("finish_reason", {}).get("type", "stop"),
                    meta=dict(req.meta or {}),
                )
            )
        return out

    async def _run(self, requests, sampling, version):
        async with self._httpx.AsyncClient() as client:
            n = requests[0].n or sampling.n
            results = await asyncio.gather(*(self._one(client, r, sampling, n, version) for r in requests))
        return [s for group in results for s in group]

    def generate(self, requests: list[GenerationRequest], sampling: SamplingConfig) -> list[Sample]:
        t0 = time.perf_counter()
        samples = asyncio.run(self._run(requests, sampling, self.policy_version))
        self._seconds += time.perf_counter() - t0
        return samples

    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        from safetensors.torch import save_file

        tmpdir = tempfile.mkdtemp(prefix="kiln-sglang-weights-")
        path = os.path.join(tmpdir, "model.safetensors")
        save_file({n: t.detach().cpu().contiguous() for n, t in named_tensors}, path)
        for url in self.urls:
            self._httpx.post(
                f"{url}/update_weights_from_disk", json={"model_path": tmpdir}, timeout=self.timeout
            ).raise_for_status()
        self.policy_version = version

    def pause(self) -> None:
        super().pause()
        for url in self.urls:
            self._httpx.post(f"{url}/pause_generation", timeout=30)

    def resume(self) -> None:
        super().resume()
        for url in self.urls:
            self._httpx.post(f"{url}/continue_generation", timeout=30)

    def stats(self) -> dict[str, float]:
        return {
            "rollout/tokens": float(self._tokens),
            "rollout/tok_per_s": self._tokens / max(self._seconds, 1e-9),
        }
