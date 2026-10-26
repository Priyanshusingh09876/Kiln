"""Gateway control plane under real HTTP load (simulated replicas, real gateway/router/canary)."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from kiln.serving.canary import SLO, CanaryController, SLOVerdict  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gw_demo", Path(__file__).resolve().parent.parent / "examples" / "04_gateway_canary_demo.py"
)
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


# ----------------------------------------------------- controller unit tests
def _feed(ctl, arm, n, lat, rng):
    for _ in range(n):
        ctl.observe(arm, lat * (0.9 + 0.2 * rng.random()))


def test_matched_control_arm_shares_and_judging():
    ctl = CanaryController(SLO(min_samples=50), ramp_steps=(0.1, 0.5, 1.0), matched_control=True)
    rng = random.Random(0)
    arms = [ctl.choose_arm(rng.random()) for _ in range(20000)]
    assert abs(arms.count("canary") / 20000 - 0.1) < 0.01
    assert abs(arms.count("control") / 20000 - 0.1) < 0.01
    # fleet traffic is slow (warm-cache mismatch irrelevant here) but must NOT be used for judgement
    _feed(ctl, "baseline", 500, 5.0, rng)
    _feed(ctl, "control", 100, 0.1, rng)
    _feed(ctl, "canary", 100, 0.1, rng)
    assert ctl.evaluate().verdict is SLOVerdict.PASS
    assert ctl.canary_share == 0.5 and ctl.control_share == 0.5
    ctl.step_index = 2
    assert ctl.control_share == 0.0  # at 100% there is no room for a control


def test_p50_ratio_catches_regression_hidden_in_noisy_tail():
    rng = random.Random(1)
    ctl = CanaryController(SLO(p99_latency_s=10.0, min_samples=100, max_latency_ratio=1.5, max_p50_ratio=1.3))
    # baseline: median 100ms but a heavy tail up to 2s; canary: median 200ms, tail no worse than baseline
    for _ in range(400):
        ctl.observe("baseline", 2.0 if rng.random() < 0.05 else 0.1)
        ctl.observe("canary", 1.5 if rng.random() < 0.05 else 0.2)
    st = ctl.evaluate()
    assert st.verdict is SLOVerdict.LATENCY and "p50" in st.detail


def test_warmup_samples_are_not_judged():
    ctl = CanaryController(SLO(min_samples=10, warmup_samples=30))
    for _ in range(30):
        ctl.observe("canary", 99.0)  # terrible during warm-up
    assert len(ctl.canary) == 0 and ctl.warmup_seen == 30


# --------------------------------------------------------- gateway behaviour
def test_healthz_does_not_advance_the_rollout():
    from fastapi.testclient import TestClient

    from kiln.serving.gateway import create_app
    from kiln.serving.router import EngineReplica

    app = create_app(
        [EngineReplica("b", "http://b"), EngineReplica("c", "http://c")],
        canary_names={"c"},
        slo=SLO(min_samples=1),
        ramp_steps=(0.1, 0.5, 1.0),
    )
    ctl = app.state.canary
    for _ in range(5):
        ctl.observe("baseline", 0.1)
        ctl.observe("canary", 0.1)
    with TestClient(app) as cl:
        for _ in range(10):
            cl.get("/healthz")
            cl.get("/canary")
    assert ctl.canary_share == 0.1, "a health probe promoted the canary"


# ------------------------------------------------------------- under load
@pytest.mark.slow
def test_prefix_routing_beats_least_loaded_on_cache_hit_rate():
    r = demo.run_routing(n_requests=400, concurrency=12)
    assert r["prefix_aware"]["replica_cache_hit_rate"] > r["least_loaded"]["replica_cache_hit_rate"] + 0.15
    assert r["prefix_aware"]["errors"] == 0 and r["least_loaded"]["errors"] == 0


@pytest.mark.slow
def test_regressed_canary_is_rolled_back_under_load_without_health_checks():
    c = demo.run_canary(1500, 12, inject_at_share=0.25, slowdown=3.0, matched_control=True)
    assert c["injected"], "canary never reached 25%: promotion path broken"
    assert c["tripped"] and c["final_share"] == 0.0
    assert c["canary_requests_after_trip"] <= 12  # only requests already in flight
