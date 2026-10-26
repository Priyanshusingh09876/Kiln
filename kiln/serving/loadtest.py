"""Reproducible load test for the gateway: simulated replicas over real HTTP.

Everything here runs on a laptop CPU. The *replicas* are simulated — each is a
small FastAPI server that mimics an LLM engine's latency, including a prefix
cache: prompt tokens already cached on that replica cost (almost) nothing,
uncached tokens cost ``prefill_us_per_token`` each, and each output token
costs ``decode_ms_per_token``. Everything else is the real thing: real
uvicorn servers on localhost ports, the real gateway app, the real router and
canary controller, real HTTP requests from a concurrent async client.

So the load test measures the *control plane's* behaviour (does routing
preserve cache locality, does a regressed canary get rolled back) with a
latency model standing in for GPUs. It does not measure GPU throughput, and the
reports it writes say so.

Used by ``examples/04_gateway_canary_demo.py`` and ``tests/test_gateway_load.py``.
"""

from __future__ import annotations

import asyncio
import random
import socket
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from kiln.serving.router import make_radix_tree


# ---------------------------------------------------------------- replicas
@dataclass
class ReplicaModel:
    """Latency model of one engine replica. Mutable at runtime via ``POST /admin``."""

    name: str
    prefill_us_per_token: float = 100.0
    decode_ms_per_token: float = 3.0
    cache_blocks: int = 1024
    block_size: int = 16
    max_concurrency: int = 16
    """Sequences the replica runs at once; more requests queue (like a full engine batch)."""
    slowdown: float = 1.0
    error_rate: float = 0.0
    seed: int = 0
    served: int = 0
    cached_tokens: int = 0
    prompt_tokens: int = 0
    tree: object = field(init=False)
    rng: random.Random = field(init=False)
    _sem: asyncio.Semaphore | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.tree = make_radix_tree(self.block_size, self.cache_blocks)
        self.rng = random.Random(self.seed)

    def semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:  # created lazily inside the server's event loop
            self._sem = asyncio.Semaphore(self.max_concurrency)
        return self._sem

    def cost_seconds(self, prompt_ids: list[int], max_tokens: int) -> tuple[float, int]:
        hit = self.tree.match(prompt_ids)
        self.tree.insert(prompt_ids)
        uncached = len(prompt_ids) - hit
        s = (
            uncached * self.prefill_us_per_token * 1e-6 + max_tokens * self.decode_ms_per_token * 1e-3
        ) * self.slowdown
        jitter = 1.0 + 0.1 * (self.rng.random() - 0.5)
        self.served += 1
        self.cached_tokens += hit
        self.prompt_tokens += len(prompt_ids)
        return s * jitter, hit


def create_replica_app(model: ReplicaModel):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"sim-replica-{model.name}")

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any]):
        prompt = str(body.get("prompt", ""))
        max_tokens = int(body.get("max_tokens", 16))
        async with model.semaphore():
            secs, hit = model.cost_seconds(list(prompt.encode("utf-8")), max_tokens)
            await asyncio.sleep(secs)
        if model.error_rate > 0 and model.rng.random() < model.error_rate:
            raise HTTPException(status_code=500, detail="simulated engine error")
        return {
            "object": "text_completion",
            "model": model.name,
            "choices": [{"index": 0, "text": "x" * max_tokens, "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": max_tokens, "cached_tokens": hit},
        }

    @app.post("/admin")
    async def admin(body: dict[str, Any]):
        for k in ("slowdown", "error_rate"):
            if k in body:
                setattr(model, k, float(body[k]))
        return {"slowdown": model.slowdown, "error_rate": model.error_rate}

    return app


# ------------------------------------------------------------ server threads
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class ServerThread:
    """Run an ASGI app with uvicorn in a background thread on a free localhost port."""

    def __init__(self, app: Any):
        import uvicorn

        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error", lifespan="off", access_log=False
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> ServerThread:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


# ---------------------------------------------------------------- workload
@dataclass
class Workload:
    """Multi-tenant chat traffic: each tenant has a long shared system prompt + short user turns."""

    tenants: int = 64
    system_prompt_tokens: int = 1024
    user_tokens: int = 48
    max_tokens: int = 16
    zipf_s: float = 1.0
    """Tenant popularity ~ 1/rank^s (s=1: the top tenant gets ~21% of 64 tenants' traffic)."""
    seed: int = 0

    def requests(self, n: int) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        sys_prompts = [
            f"[tenant {t}] "
            + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(self.system_prompt_tokens))
            for t in range(self.tenants)
        ]
        weights = [1.0 / (k + 1) ** self.zipf_s for k in range(self.tenants)]
        out = []
        for _ in range(n):
            t = rng.choices(range(self.tenants), weights)[0]
            user = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(self.user_tokens))
            out.append(
                {"prompt": sys_prompts[t] + "\nUSER: " + user, "max_tokens": self.max_tokens, "tenant": t}
            )
        return out


@dataclass
class Result:
    latency_s: float
    ok: bool
    replica: str | None
    arm: str | None
    prefix_hit_tokens: int
    t_start: float


async def _drive(url: str, reqs: list[dict[str, Any]], concurrency: int, on_progress=None) -> list[Result]:
    import httpx

    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []
    t_origin = time.perf_counter()
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=concurrency * 2)) as client:

        async def one(i: int, body: dict[str, Any]) -> None:
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(
                        f"{url}/v1/completions", json={k: v for k, v in body.items() if k != "tenant"}
                    )
                    ok = r.status_code == 200
                    meta = r.json().get("kiln", {}) if ok else {}
                except Exception:
                    ok, meta = False, {}
                results.append(
                    Result(
                        time.perf_counter() - t0,
                        ok,
                        meta.get("replica"),
                        meta.get("arm"),
                        int(meta.get("prefix_hit_tokens", 0)),
                        t0 - t_origin,
                    )
                )
                if on_progress is not None:
                    await on_progress(i)

        await asyncio.gather(*(one(i, b) for i, b in enumerate(reqs)))
    return results


def drive(url: str, reqs: list[dict[str, Any]], concurrency: int = 32, on_progress=None) -> list[Result]:
    return asyncio.run(_drive(url, reqs, concurrency, on_progress))


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)]


def summarize(results: list[Result], wall_s: float, replicas: list[ReplicaModel]) -> dict[str, float]:
    lat = [r.latency_s for r in results if r.ok]
    prompt = sum(m.prompt_tokens for m in replicas)
    cached = sum(m.cached_tokens for m in replicas)
    return {
        "requests": len(results),
        "errors": sum(1 for r in results if not r.ok),
        "throughput_rps": len(results) / max(wall_s, 1e-9),
        "latency_p50_ms": pct(lat, 50) * 1e3,
        "latency_p99_ms": pct(lat, 99) * 1e3,
        "latency_mean_ms": (statistics.fmean(lat) * 1e3) if lat else float("nan"),
        "replica_cache_hit_rate": cached / max(prompt, 1),
        "max_replica_share": max(m.served for m in replicas) / max(sum(m.served for m in replicas), 1),
    }
