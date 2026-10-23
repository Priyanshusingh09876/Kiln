"""Router, scheduler/KV allocator, canary controller and gateway."""

from __future__ import annotations

import random

import pytest

from kiln.serving import (
    SLO,
    CanaryController,
    EngineReplica,
    KVBlockAllocator,
    PrefixRouter,
    Request,
    Scheduler,
    SchedulerConfig,
    SLOVerdict,
    simulate,
)
from kiln.serving.router import BlockRadixTree


# ------------------------------------------------------------------ router
def test_radix_tree_prefix_match_and_lru_eviction():
    t = BlockRadixTree(block_size=4, capacity_blocks=3)
    a = list(range(12))  # 3 blocks
    assert t.insert(a) == 3 and t.match(a) == 12
    assert t.match(a[:8] + [99, 99, 99, 99]) == 8
    assert t.match([5, 6, 7, 8]) == 0
    t.insert([100, 101, 102, 103])  # over capacity -> evict LRU leaf
    assert t.size_blocks <= 3


def test_router_prefers_replica_with_prefix_hit():
    reps = [EngineReplica("a", "http://a"), EngineReplica("b", "http://b")]
    r = PrefixRouter(reps, load_weight=0.0)
    sys_prompt = list(range(64))
    d1 = r.route(sys_prompt + [1, 2, 3])
    r.release(d1.replica)
    d2 = r.route(sys_prompt + [9, 9, 9])
    assert d2.replica is d1.replica and d2.matched_tokens == 64 and d2.reason == "prefix-hit"
    assert r.stats()[d1.replica.name]["hit_rate"] > 0


def test_router_balances_load_under_imbalance_cap():
    reps = [EngineReplica("a", "http://a"), EngineReplica("b", "http://b")]
    r = PrefixRouter(reps, max_imbalance=2)
    p = list(range(32))
    chosen = [r.route(p).replica.name for _ in range(10)]  # never released -> load piles up
    assert chosen.count("a") >= 3 and chosen.count("b") >= 3
    assert abs(reps[0].in_flight - reps[1].in_flight) <= 3


def test_router_skips_unhealthy():
    reps = [EngineReplica("a", "http://a"), EngineReplica("b", "http://b")]
    r = PrefixRouter(reps)
    r.mark("a", False)
    assert all(r.route([1, 2]).replica.name == "b" for _ in range(5))
    r.mark("b", False)
    with pytest.raises(RuntimeError):
        r.route([1])


# --------------------------------------------------------------- scheduler
def test_kv_allocator():
    kv = KVBlockAllocator(num_blocks=4, block_size=16)
    assert kv.blocks_for(17) == 2
    b = kv.allocate(3)
    assert kv.free_blocks == 1 and kv.peak_used == 3
    with pytest.raises(MemoryError):
        kv.allocate(2)
    kv.free(b)
    assert kv.free_blocks == 4


def test_scheduler_admits_decodes_and_finishes():
    s = Scheduler(SchedulerConfig(num_blocks=64, block_size=16, max_num_seqs=8, max_num_batched_tokens=64))
    s.add(Request(0.0, 0, prompt_len=20, max_new=5))
    st = s.step()
    assert st.admitted == 1 and st.prefill_tokens == 20
    for _ in range(10):
        s.step()
    assert len(s.finished) == 1
    r = s.finished[0]
    assert r.generated == 5 and r.ttft is not None and r.e2e >= r.ttft
    assert s.kv.used_blocks == 0  # blocks returned


def test_chunked_prefill_splits_long_prompt():
    s = Scheduler(SchedulerConfig(num_blocks=1024, max_num_batched_tokens=100, chunked_prefill=True))
    s.add(Request(0.0, 0, prompt_len=250, max_new=3))
    st1, st2, st3 = s.step(), s.step(), s.step()
    assert st1.prefill_tokens == 100 and st2.prefill_tokens == 100 and st3.prefill_tokens == 50
    assert s.running[0].prefill_done


def test_long_prompt_does_not_starve_decoders_with_chunking():
    cfg = SchedulerConfig(num_blocks=1024, max_num_batched_tokens=64, chunked_prefill=True, max_num_seqs=8)
    s = Scheduler(cfg)
    s.add(Request(0.0, 0, prompt_len=16, max_new=8))
    s.step()  # short request is now decoding
    s.add(Request(0.0, 1, prompt_len=1000, max_new=1))
    st = s.step()
    assert (
        st.decode_seqs == 1
    )  # the running decoder still progressed while the long prompt prefills in chunks


def test_preemption_evicts_youngest_and_recomputes():
    cfg = SchedulerConfig(
        num_blocks=8, block_size=16, max_num_seqs=8, max_num_batched_tokens=10_000, decode_reserve_blocks=0
    )
    s = Scheduler(cfg)
    s.add(Request(0.0, 0, prompt_len=24, max_new=40))  # 2 blocks
    s.add(Request(0.0, 1, prompt_len=24, max_new=40))  # 2 blocks
    s.add(Request(0.0, 2, prompt_len=24, max_new=40))  # 2 blocks -> 2 spare, not enough for 3 growing seqs
    s.step()
    assert len(s.running) == 3 and s.kv.free_blocks == 2
    for _ in range(40):
        s.step()
    assert s.total_preemptions > 0
    victims = [r for r in s.finished + s.running + list(s.waiting) if r.preemptions > 0]
    assert all(v.rid != 0 for v in victims)  # oldest admitted is never the first victim


def test_simulation_saturates_gracefully():
    low = simulate(SchedulerConfig(num_blocks=2048), num_requests=200, arrival_rps=2, seed=1)
    high = simulate(SchedulerConfig(num_blocks=2048), num_requests=200, arrival_rps=200, seed=1)
    assert low.requests == high.requests == 200
    assert low.ttft_p99 < high.ttft_p99
    assert high.throughput_rps < 200 and 0 < high.peak_kv_util <= 1


# ------------------------------------------------------------------ canary
def _feed(ctl, arm, n, lat, err=0.0, q=None, rng=None):
    rng = rng or random.Random(0)
    for _ in range(n):
        ctl.observe(arm, lat * (0.8 + 0.4 * rng.random()), ok=rng.random() >= err, quality=q)


def test_canary_promotes_through_ramp_when_slo_holds():
    ctl = CanaryController(SLO(p99_latency_s=0.5, min_samples=50), ramp_steps=(0.1, 0.5, 1.0))
    assert ctl.canary_share == 0.1
    assert ctl.evaluate().verdict is SLOVerdict.INSUFFICIENT
    for _ in range(2):
        _feed(ctl, "baseline", 100, 0.2)
        _feed(ctl, "canary", 100, 0.2)
        assert ctl.evaluate().verdict is SLOVerdict.PASS
    assert ctl.fully_promoted and ctl.canary_share == 1.0


def test_canary_rollback_on_latency_and_sticky():
    ctl = CanaryController(SLO(p99_latency_s=0.5, min_samples=50))
    _feed(ctl, "baseline", 100, 0.2)
    _feed(ctl, "canary", 100, 0.6)
    st = ctl.evaluate()
    assert st.verdict is SLOVerdict.LATENCY and ctl.tripped and ctl.canary_share == 0.0
    _feed(ctl, "canary", 500, 0.1)
    assert ctl.evaluate().tripped  # stays tripped until reset
    ctl.reset()
    assert not ctl.tripped and ctl.canary_share == 0.01


def test_canary_relative_latency_error_rate_and_quality():
    ctl = CanaryController(SLO(p99_latency_s=10, max_latency_ratio=1.2, min_samples=50))
    _feed(ctl, "baseline", 100, 0.1)
    _feed(ctl, "canary", 100, 0.2)
    assert ctl.evaluate().verdict is SLOVerdict.LATENCY
    ctl = CanaryController(SLO(p99_latency_s=10, max_latency_ratio=None, max_error_rate=0.01, min_samples=50))
    _feed(ctl, "canary", 200, 0.1, err=0.2)
    assert ctl.evaluate().verdict is SLOVerdict.ERRORS
    ctl = CanaryController(
        SLO(p99_latency_s=10, max_latency_ratio=None, max_quality_drop=0.05, min_samples=50)
    )
    _feed(ctl, "baseline", 100, 0.1, q=0.9)
    _feed(ctl, "canary", 100, 0.1, q=0.7)
    assert ctl.evaluate().verdict is SLOVerdict.QUALITY


def test_choose_arm_respects_share():
    ctl = CanaryController(SLO(), ramp_steps=(0.25,))
    arms = [ctl.choose_arm(random.Random(1).random()) for _ in range(1)]
    assert arms[0] in ("canary", "baseline")
    rng = random.Random(3)
    frac = sum(ctl.choose_arm(rng.random()) == "canary" for _ in range(4000)) / 4000
    assert 0.2 < frac < 0.3


def test_gateway_routes_and_reports_metrics(monkeypatch):
    pytest.importorskip("fastapi")
    import httpx
    from fastapi.testclient import TestClient

    from kiln.serving.gateway import create_app

    calls = []

    async def fake_post(self, url, json=None):
        calls.append(url)
        return httpx.Response(200, json={"choices": [{"text": "ok"}]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    app = create_app([EngineReplica("a", "http://a"), EngineReplica("b", "http://b")])
    with TestClient(app) as c:
        for _ in range(4):
            assert (
                c.post("/v1/completions", json={"prompt": "same prefix " * 10, "max_tokens": 1}).status_code
                == 200
            )
        assert len({u for u in calls}) == 1  # sticky to one replica thanks to the prefix cache
        m = c.get("/metrics").text
        assert "kiln_gateway_requests_total 4" in m and "kiln_replica_hit_rate" in m
        assert c.get("/healthz").json()["ok"]
