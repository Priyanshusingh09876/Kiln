"""OpenAI-compatible gateway in front of N engine replicas.

``POST /v1/completions`` -> tokenize (for routing only) -> :class:`PrefixRouter`
picks a replica -> proxy the request -> record latency into the
:class:`CanaryController` if a canary is configured. ``/metrics`` exposes
Prometheus text so the same dashboards work in k8s and on a laptop.

Canary evaluation is driven by canary traffic: every ``eval_every`` canary
observations the controller re-evaluates and may promote or trip. Health
checks (``/healthz``) and ``/canary`` are read-only, so a k8s liveness probe
can never advance a rollout.

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
    control_names: set[str] | None = None,
    tokenizer: Any = None,
    router_kwargs: dict[str, Any] | None = None,
    ramp_steps: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 1.0),
    eval_every: int = 50,
    seed: int | None = None,
):
    try:
        import httpx
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import PlainTextResponse
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'kiln-rl[reward]' for the gateway (fastapi/httpx)") from e

    canary_names = canary_names or set()
    control_names = control_names or set()
    router_kwargs = router_kwargs or {}
    baseline = [r for r in replicas if r.name not in canary_names | control_names]
    canary = [r for r in replicas if r.name in canary_names]
    control = [r for r in replicas if r.name in control_names]
    router_base = PrefixRouter(baseline, **router_kwargs)
    router_canary = PrefixRouter(canary, **router_kwargs) if canary else None
    router_control = PrefixRouter(control, **router_kwargs) if control else None
    ctl = (
        CanaryController(slo or SLO(), ramp_steps=ramp_steps, matched_control=bool(control))
        if canary
        else None
    )
    rng = random.Random(seed)
    client = httpx.AsyncClient(
        timeout=120, limits=httpx.Limits(max_connections=512, max_keepalive_connections=128)
    )
    app = FastAPI(title="kiln-gateway", version="0.3.0")
    counters: dict[str, Any] = {"requests": 0, "errors": 0, "latency_sum": 0.0, "canary_obs": 0}
    timeline: list[dict[str, Any]] = []

    def _ids(prompt: str) -> list[int]:
        if tokenizer is not None:
            return tokenizer.encode(prompt)
        return list(prompt.encode("utf-8"))  # byte-level is a fine routing key

    def _maybe_evaluate() -> None:
        assert ctl is not None
        counters["canary_obs"] += 1
        if counters["canary_obs"] % eval_every:
            return
        st = ctl.evaluate()
        timeline.append(
            {
                "request": counters["requests"],
                "share": st.share,
                "verdict": st.verdict.value,
                "tripped": st.tripped,
                "canary_p99_s": st.canary_p99,
                "baseline_p99_s": st.baseline_p99,
                "detail": st.detail,
            }
        )

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any]):
        prompt = body.get("prompt", "")
        arm = "baseline"
        router = router_base
        if ctl is not None and router_canary is not None:
            arm = ctl.choose_arm(rng.random())
            router = {"canary": router_canary, "control": router_control}.get(arm) or router_base
        decision = router.route(_ids(prompt if isinstance(prompt, str) else str(prompt)))
        t0 = time.perf_counter()
        ok = True
        try:
            resp = await client.post(f"{decision.replica.url}/v1/completions", json=body)
            resp.raise_for_status()
            out = resp.json()
            out.setdefault("kiln", {}).update(
                {"replica": decision.replica.name, "arm": arm, "prefix_hit_tokens": decision.matched_tokens}
            )
            return out
        except Exception as e:
            ok = False
            counters["errors"] += 1
            raise HTTPException(status_code=502, detail=f"{decision.replica.name}: {e}") from e
        finally:
            dt = time.perf_counter() - t0
            router.release(decision.replica)
            counters["requests"] += 1
            counters["latency_sum"] += dt
            if ctl is not None:
                ctl.observe(arm, dt, ok)
                if arm == "canary":
                    _maybe_evaluate()

    @app.get("/healthz")
    async def healthz():
        last = ctl.history[-1].__dict__ if ctl is not None and ctl.history else None
        return {"ok": True, "replicas": router_base.stats(), "canary": last}

    @app.get("/canary")
    async def canary_state():
        if ctl is None:
            return {"enabled": False}
        return {"enabled": True, "share": ctl.canary_share, "tripped": ctl.tripped, "timeline": timeline}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = [
            f"kiln_gateway_requests_total {counters['requests']}",
            f"kiln_gateway_errors_total {counters['errors']}",
            f"kiln_gateway_latency_seconds_sum {counters['latency_sum']:.6f}",
        ]
        all_stats = {**router_base.stats()}
        for extra in (router_canary, router_control):
            if extra is not None:
                all_stats.update(extra.stats())
        for name, s in all_stats.items():
            for k, v in s.items():
                lines.append(f'kiln_replica_{k}{{replica="{name}"}} {v}')
        if ctl is not None:
            lines.append(f"kiln_canary_share {ctl.canary_share}")
            lines.append(f"kiln_canary_tripped {int(ctl.tripped)}")
        return "\n".join(lines) + "\n"

    app.state.router = router_base
    app.state.router_canary = router_canary
    app.state.canary = ctl
    app.state.timeline = timeline
    app.state.counters = counters
    return app
