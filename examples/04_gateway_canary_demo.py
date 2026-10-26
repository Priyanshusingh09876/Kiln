"""Gateway under load: prefix-cache-aware routing and SLO-gated canary rollback.

    python examples/04_gateway_canary_demo.py            # ~1 minute on a laptop
    python examples/04_gateway_canary_demo.py --quick    # ~15 s

Starts simulated engine replicas and the real Kiln gateway as HTTP servers on
localhost, drives multi-tenant traffic through the gateway with a concurrent
client, and writes ``docs/benchmarks/gateway_demo.{md,json}`` (+ a PNG if
matplotlib is installed).

Scenario 1 — routing. Same traffic, same 4 replicas (cold caches each time),
two routing policies: least-loaded (what a generic L7 load balancer does) vs.
Kiln's prefix-aware router. Measures replica prefix-cache hit rate and latency.

Scenario 2 — canary. 3 baseline replicas + 1 canary on a 10% -> 25% -> 50% ->
100% ramp (10% first). The canary is healthy at first and gets promoted; once it reaches
25% of traffic it is made 3x slower (a simulated bad build). The controller
should trip and send its share to 0 — automatically, with no health check or
operator involved.

Replica latency is a *model* (see kiln/serving/loadtest.py); the gateway,
router, canary controller and HTTP stack are real. Numbers describe control-plane
behaviour, not GPU throughput.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from kiln.serving.canary import SLO
from kiln.serving.gateway import create_app
from kiln.serving.loadtest import ReplicaModel, ServerThread, Workload, create_replica_app, drive, summarize
from kiln.serving.router import EngineReplica

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"


def _start_replicas(models: list[ReplicaModel]):
    servers = [ServerThread(create_replica_app(m)) for m in models]
    for s in servers:
        s.__enter__()
    return servers


def _stop(servers) -> None:
    for s in servers:
        s.__exit__(None, None, None)


def run_routing(
    n_requests: int, concurrency: int, n_replicas: int = 4, cache_blocks: int = 1024, seed: int = 0
) -> dict:
    """Least-loaded vs prefix-aware routing on identical traffic and cold replicas."""
    wl = Workload(seed=seed)
    reqs = wl.requests(n_requests)
    out = {}
    for policy, kwargs in (("least_loaded", {"cache_weight": 0.0}), ("prefix_aware", {})):
        models = [ReplicaModel(f"r{i}", cache_blocks=cache_blocks, seed=seed + i) for i in range(n_replicas)]
        servers = _start_replicas(models)
        reps = [EngineReplica(m.name, s.url) for m, s in zip(models, servers)]
        try:
            with ServerThread(create_app(reps, router_kwargs=kwargs)) as gw:
                t0 = time.perf_counter()
                res = drive(gw.url, reqs, concurrency)
                out[policy] = summarize(res, time.perf_counter() - t0, models)
                out[policy]["per_replica_served"] = {m.name: m.served for m in models}
        finally:
            _stop(servers)
    return out


def run_canary(
    n_requests: int,
    concurrency: int,
    inject_at_share: float | None = 0.25,
    slowdown: float = 3.0,
    matched_control: bool = True,
    seed: int = 0,
) -> dict:
    """Canary on a ramp; optionally regressed once it reaches ``inject_at_share``.

    ``matched_control=False`` judges the canary against the warm fleet (the naive setup);
    ``True`` adds a fresh control replica of the current build that gets the same traffic
    share as the canary, and judges against that.
    """
    wl = Workload(seed=seed + 1)
    reqs = wl.requests(n_requests)
    models = [ReplicaModel(f"base{i}", seed=seed + i) for i in range(3)] + [
        ReplicaModel("canary0", seed=seed + 99)
    ]
    if matched_control:
        models.append(ReplicaModel("control0", seed=seed + 98))
    servers = _start_replicas(models)
    reps = [EngineReplica(m.name, s.url) for m, s in zip(models, servers)]
    canary_model = models[3]
    slo = SLO(
        p99_latency_s=10.0,
        max_latency_ratio=1.5,
        max_p50_ratio=1.3,
        max_error_rate=0.02,
        max_quality_drop=None,
        min_samples=60,
    )
    app = create_app(
        reps,
        canary_names={"canary0"},
        control_names={"control0"} if matched_control else None,
        slo=slo,
        ramp_steps=(0.1, 0.25, 0.5, 1.0),
        eval_every=20,
        seed=seed,
    )
    ctl = app.state.canary
    injected: dict = {}
    share_trace: list[tuple[int, float]] = []

    async def on_progress(i: int) -> None:
        share_trace.append((len(share_trace), ctl.canary_share))
        if inject_at_share is not None and not injected and ctl.canary_share >= inject_at_share:
            canary_model.slowdown = slowdown
            injected.update({"request_index": len(share_trace), "share": ctl.canary_share})

    try:
        with ServerThread(app) as gw:
            t0 = time.perf_counter()
            res = drive(gw.url, reqs, concurrency, on_progress=on_progress)
            wall = time.perf_counter() - t0
    finally:
        _stop(servers)

    trip = next((e for e in app.state.timeline if e["tripped"]), None)
    after_trip = res[trip["request"] :] if trip else []
    return {
        "matched_control": matched_control,
        "promotions": [e["share"] for e in app.state.timeline if e["detail"].startswith("promoted")],
        "summary": summarize(res, wall, models),
        "injected": injected,
        "tripped": ctl.tripped,
        "final_share": ctl.canary_share,
        "trip_event": trip,
        "requests_to_canary_after_injection": sum(
            1 for r in res[injected.get("request_index", len(res)) :] if r.arm == "canary"
        ),
        "canary_requests_after_trip": sum(1 for r in after_trip if r.arm == "canary"),
        "timeline": app.state.timeline,
        "share_trace": share_trace[:: max(1, len(share_trace) // 400)],
    }


def _canary_outcome(c: dict) -> str:
    if c["tripped"]:
        t = c["trip_event"]
        return f"rolled back at request {t['request']} (`{t['detail']}`)"
    return f"promoted to {c['final_share']:.0%}"


def write_report(routing: dict, canaries: dict, args) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "requests_routing": args.requests,
        "requests_canary": args.canary_requests,
        "concurrency": args.concurrency,
    }
    (OUT / "gateway_demo.json").write_text(
        json.dumps({"meta": meta, "routing": routing, "canary": canaries}, indent=2, default=float)
    )
    ll, pa = routing["least_loaded"], routing["prefix_aware"]
    lines = [
        "# Gateway under load: routing and canary rollback",
        "",
        f"Generated by `python examples/04_gateway_canary_demo.py` on {meta['date']} ({meta['machine']}).",
        "Replica latency is simulated (prefix cache + per-token cost + per-replica concurrency limit, see",
        "`kiln/serving/loadtest.py`); the gateway, router, canary controller and HTTP stack are the real code.",
        "These numbers describe control-plane behaviour, **not GPU throughput**.",
        "",
        f"## 1. Routing: least-loaded vs prefix-aware ({args.requests} requests, concurrency {args.concurrency}, 4 replicas)",
        "",
        "Workload: 64 tenants with Zipf(1.0) popularity, each with its own 1024-token system prompt; each replica's",
        "prefix cache holds about 16 tenants, so the hot set does not fit on every replica and locality matters.",
        "Each policy starts from cold replicas and gets identical traffic.",
        "",
        "| policy | replica prefix-cache hit rate | p50 latency | p99 latency | throughput | busiest replica share |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in (("least-loaded", ll), ("prefix-aware (Kiln)", pa)):
        lines.append(
            f"| {name} | {r['replica_cache_hit_rate']:.1%} | {r['latency_p50_ms']:.0f} ms | {r['latency_p99_ms']:.0f} ms "
            f"| {r['throughput_rps']:.0f} req/s | {r['max_replica_share']:.0%} |"
        )
    d50 = pa["latency_p50_ms"] / ll["latency_p50_ms"] - 1
    d99 = pa["latency_p99_ms"] / ll["latency_p99_ms"] - 1
    dt = pa["throughput_rps"] / ll["throughput_rps"] - 1

    def _chg(d: float) -> str:
        return f"{'down' if d < 0 else 'up'} {abs(d):.0%}"

    lines += [
        "",
        f"Prefix-aware routing moved the replica cache hit rate from {ll['replica_cache_hit_rate']:.1%} to "
        f"{pa['replica_cache_hit_rate']:.1%}: p50 latency {_chg(d50)}, p99 latency {_chg(d99)}, throughput {_chg(dt)}. "
        f"The cost is balance: the busiest replica handled {pa['max_replica_share']:.0%} of traffic "
        f"(least-loaded: {ll['max_replica_share']:.0%}, perfect: 25%). `PrefixRouter(max_imbalance=...)` trades one for the other.",
        "",
        f"## 2. Canary rollouts ({args.canary_requests} requests each)",
        "",
        "Ramp 10% → 25% → 50% → 100%. SLO: canary p50 ≤ 1.3× baseline p50, canary p99 ≤ 1.5× baseline p99,",
        "error rate ≤ 2%, ≥ 60 canary samples per step before any decision. Three runs:",
        "",
        "| run | judged against | canary | expected | outcome |",
        "|---|---|---|---|---|",
    ]
    rows = [
        ("A", "warm fleet", "healthy", "promote to 100%", canaries["A_fleet_healthy"]),
        ("B", "matched control", "healthy", "promote to 100%", canaries["B_control_healthy"]),
        (
            "C",
            "matched control",
            f"{args.slowdown:g}× slower from 25%",
            "roll back",
            canaries["C_control_regressed"],
        ),
    ]
    for run, against, canary, expected, c in rows:
        ok = (c["tripped"] and expected == "roll back") or (
            not c["tripped"] and expected != "roll back" and c["final_share"] == 1.0
        )
        lines.append(
            f"| {run} | {against} | {canary} | {expected} | {'✅' if ok else '❌'} {_canary_outcome(c)} |"
        )
    a = canaries["A_fleet_healthy"]
    c = canaries["C_control_regressed"]
    lines += [
        "",
        "**Run A is a false positive, and it is the point of the comparison.** A new replica starts with a cold",
        "prefix cache and, at a 10% share, sees a random sample of all 64 tenants, while each fleet replica owns a",
        "cache-friendly slice thanks to prefix routing. Judged against the fleet, a perfectly healthy build looks",
        "slower and gets rolled back."
        if a["tripped"]
        else "**Run A did not trip this time**; judging against the warm fleet is still biased (see ADR 04).",
        "",
        "Run B fixes it the way Netflix's Kayenta does: a *control* replica of the current build starts at the",
        "same time as the canary and gets exactly the same traffic share, so both are equally cold and see the",
        "same request mix; the canary is judged only against the control.",
        "",
    ]
    if c["injected"]:
        lines.append(
            f"Run C: the canary was made {args.slowdown:g}× slower at request {c['injected']['request_index']} "
            f"(share {c['injected']['share']:.0%}). "
            + (
                f"The controller rolled it back at request {c['trip_event']['request']}; canary requests completing "
                f"after the rollback: **{c['canary_requests_after_trip']}** (requests already in flight at that moment; "
                f"client concurrency is {args.concurrency})."
                if c["tripped"]
                else "The controller did **not** roll it back (see JSON)."
            )
        )
        lines += [
            "",
            "Run C decision log:",
            "",
            "| at request | canary share | verdict | detail |",
            "|---|---|---|---|",
        ]
        for e in c["timeline"]:
            if e["verdict"] != "insufficient_data":
                lines.append(f"| {e['request']} | {e['share']:.0%} | {e['verdict']} | {e['detail']} |")
    md = OUT / "gateway_demo.md"
    md.write_text("\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.2))
        for key, label, style in (
            ("A_fleet_healthy", "A: healthy, vs warm fleet", ":"),
            ("B_control_healthy", "B: healthy, vs matched control", "-"),
            ("C_control_regressed", "C: regressed at 25%, vs matched control", "--"),
        ):
            tr = canaries[key]["share_trace"]
            ax.step([i for i, _ in tr], [s for _, s in tr], where="post", ls=style, label=label)
        if c["injected"]:
            ax.axvline(c["injected"]["request_index"], c="tab:red", lw=0.8, alpha=0.6)
        ax.set_xlabel("request")
        ax.set_ylabel("canary traffic share")
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="center right", fontsize=7)
        ax.set_title("SLO-gated canary rollouts")
        fig.tight_layout()
        fig.savefig(OUT / "gateway_canary.png", dpi=130)
    except ImportError:
        pass
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--canary-requests", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--slowdown", type=float, default=3.0)
    args = ap.parse_args()
    if args.quick:
        args.requests, args.canary_requests = 800, 2500
    routing = run_routing(args.requests, args.concurrency)
    n, k = args.canary_requests, args.concurrency
    canaries = {
        "A_fleet_healthy": run_canary(n, k, inject_at_share=None, matched_control=False),
        "B_control_healthy": run_canary(n, k, inject_at_share=None, matched_control=True),
        "C_control_regressed": run_canary(
            n, k, inject_at_share=0.25, slowdown=args.slowdown, matched_control=True
        ),
    }
    md = write_report(routing, canaries, args)
    print(md.read_text())


if __name__ == "__main__":
    main()
