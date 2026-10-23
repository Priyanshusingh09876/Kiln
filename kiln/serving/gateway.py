"""OpenAI-compatible gateway in front of N engine replicas.

``POST /v1/completions`` -> tokenize (for routing only) -> :class:`PrefixRouter`
picks a replica -> proxy the request -> record latency into the
:class:`CanaryController` if a canary is configured. ``/metrics`` exposes
Prometheus text so the same dashboards work in k8s and on a laptop.

Run with ``kiln gateway --replica name=http://host:8000 ...``.
"""

from __future__ import annotations

import random
import time
from typing import Any

from kiln.serving.canary import SLO, CanaryController
from kiln.serving.router import EngineReplica, PrefixRouter


def create_app(
    replicas: list[EngineReplica],
    canary_names: set[str] | None = None,
    slo: SLO | None = None,
    tokenizer: Any = None,
):
    try:
        import httpx
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'kiln-rl[reward]' for the gateway (fastapi/httpx)") from e

    canary_names = canary_names or set()
    baseline = [r for r in replicas if r.name not in canary_names]
    canary = [r for r in replicas if r.name in canary_names]
    router_base = PrefixRouter(baseline)
    router_canary = PrefixRouter(canary) if canary else None
    ctl = CanaryController(slo or SLO()) if canary else None
    client = httpx.AsyncClient(timeout=120)
    app = FastAPI(title="kiln-gateway", version="0.3.0")
    counters = {"requests": 0, "errors": 0, "latency_sum": 0.0}

    def _ids(prompt: str) -> list[int]:
        if tokenizer is not None:
            return tokenizer.encode(prompt)
        return list(prompt.encode("utf-8"))  # byte-level is a fine routing key

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any]):
        prompt = body.get("prompt", "")
        arm = "baseline"
        router = router_base
        if ctl is not None and router_canary is not None:
            arm = ctl.choose_arm(random.random())
            router = router_canary if arm == "canary" else router_base
        decision = router.route(_ids(prompt if isinstance(prompt, str) else str(prompt)))
        t0 = time.perf_counter()
        ok = True
        try:
            resp = await client.post(f"{decision.replica.url}/v1/completions", json=body)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            ok = False
            counters["errors"] += 1
            raise HTTPException(status_code=502, detail=str(e)) from e
        finally:
            dt = time.perf_counter() - t0
            router.release(decision.replica)
            counters["requests"] += 1
            counters["latency_sum"] += dt
            if ctl is not None:
                ctl.observe(arm, dt, ok)

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "replicas": router_base.stats(),
            "canary": None if ctl is None else ctl.evaluate().__dict__,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = [
            f"kiln_gateway_requests_total {counters['requests']}",
            f"kiln_gateway_errors_total {counters['errors']}",
            f"kiln_gateway_latency_seconds_sum {counters['latency_sum']:.6f}",
        ]
        for name, s in {**router_base.stats(), **(router_canary.stats() if router_canary else {})}.items():
            for k, v in s.items():
                lines.append(f'kiln_replica_{k}{{replica="{name}"}} {v}')
        if ctl is not None:
            lines.append(f"kiln_canary_share {ctl.canary_share}")
            lines.append(f"kiln_canary_tripped {int(ctl.tripped)}")
        return "\n".join(lines) + "\n"

    app.state.router = router_base
    app.state.canary = ctl
    return app
