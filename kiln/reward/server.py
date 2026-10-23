"""Reward server (FastAPI) + client.

Why a server at all? In a disaggregated cluster the reward computation
(rule verifiers that shell out to Python, or a GPU reward model) should not
block the trainer's or the rollout engines' GPUs. Rollout workers POST a
batch of ``{response, answer}`` and get rewards back; the server batches
reward-model forward passes and runs rule verifiers in a thread pool.

Endpoints
---------
``POST /score``   body: ``{"items": [{"response": str, "answer": str|null, "meta": {...}}, ...]}``
                  returns ``{"rewards": [float], "breakdown": [{name: float}]}``
``GET  /healthz`` liveness
``GET  /metrics`` plain-text counters (Prometheus exposition format)
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from kiln.config.schema import RewardConfig
from kiln.reward import rules

try:  # request/response models live at module scope so FastAPI can resolve the annotations
    from pydantic import BaseModel as _BaseModel

    class ScoreItem(_BaseModel):
        response: str
        answer: str | None = None
        prompt: str = ""
        meta: dict[str, Any] = {}

    class ScoreRequest(_BaseModel):
        items: list[ScoreItem]

    class ScoreResponse(_BaseModel):
        rewards: list[float]
        breakdown: list[dict[str, float]]

except ImportError:  # pragma: no cover
    ScoreItem = ScoreRequest = ScoreResponse = None  # type: ignore


@dataclass
class _Counters:
    requests: int = 0
    items: int = 0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, items: int, seconds: float) -> None:
        with self.lock:
            self.requests += 1
            self.items += items
            self.seconds += seconds

    def prometheus(self) -> str:
        return (
            f"kiln_reward_requests_total {self.requests}\n"
            f"kiln_reward_items_total {self.items}\n"
            f"kiln_reward_seconds_total {self.seconds:.6f}\n"
        )


class RewardScorer:
    """Backend-agnostic scorer used both in-process and behind the HTTP server."""

    def __init__(self, cfg: RewardConfig, workers: int = 8):
        self.cfg = cfg
        self.fns = [(f.name, f.weight, f.params) for f in cfg.functions]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.counters = _Counters()
        self._model = None
        if cfg.mode == "model":
            from kiln.reward.model_reward import RewardModelScorer

            self._model = RewardModelScorer(cfg.model_name_or_path or "", batch_size=cfg.batch_size)

    def score_one(self, response: str, answer: str | None) -> tuple[float, dict[str, float]]:
        return rules.score(response, answer, self.fns)

    def score_batch(self, items: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, float]]]:
        t0 = time.perf_counter()
        if self._model is not None:
            rewards = self._model.score([(it.get("prompt", ""), it["response"]) for it in items])
            breakdown = [{"reward_model": r} for r in rewards]
        else:
            futs = [self.pool.submit(self.score_one, it["response"], it.get("answer")) for it in items]
            results = [f.result() for f in futs]
            rewards = [r for r, _ in results]
            breakdown = [b for _, b in results]
        self.counters.add(len(items), time.perf_counter() - t0)
        return rewards, breakdown


def create_app(cfg: RewardConfig):
    try:
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'kiln-rl[reward]'") from e

    scorer = RewardScorer(cfg)
    app = FastAPI(title="kiln-reward", version="0.3.0")

    @app.post("/score", response_model=ScoreResponse)
    async def score(req: ScoreRequest) -> ScoreResponse:
        loop = asyncio.get_running_loop()
        rewards, breakdown = await loop.run_in_executor(
            None, scorer.score_batch, [i.model_dump() for i in req.items]
        )
        return ScoreResponse(rewards=rewards, breakdown=breakdown)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "functions": [f.name for f in cfg.functions], "mode": cfg.mode}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return scorer.counters.prometheus()

    app.state.scorer = scorer
    return app


class RewardClient:
    """Synchronous client with retries; used by rollout workers."""

    def __init__(self, url: str, timeout_s: float = 30.0, retries: int = 3, batch_size: int = 64):
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)
        self.retries = retries
        self.batch_size = batch_size

    def score(self, items: list[dict[str, Any]]) -> tuple[list[float], list[dict[str, float]]]:
        rewards: list[float] = []
        breakdown: list[dict[str, float]] = []
        for s in range(0, len(items), self.batch_size):
            chunk = items[s : s + self.batch_size]
            for attempt in range(self.retries):
                try:
                    r = self.client.post(f"{self.url}/score", json={"items": chunk})
                    r.raise_for_status()
                    body = r.json()
                    rewards.extend(body["rewards"])
                    breakdown.extend(body["breakdown"])
                    break
                except Exception:
                    if attempt == self.retries - 1:
                        raise
                    time.sleep(0.5 * 2**attempt)
        return rewards, breakdown


def serve(cfg: RewardConfig, host: str = "0.0.0.0", port: int = 8008) -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")
