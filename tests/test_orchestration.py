"""Experience buffer, staleness controller, rollout worker and weight sync."""

from __future__ import annotations

import threading
import time

import torch

from kiln.config.schema import KilnConfig
from kiln.data.prompts import ByteTokenizer, PromptDataset
from kiln.model import TinyConfig, TinyLM
from kiln.orchestration import ExperienceBuffer, RolloutWorker, StalenessController
from kiln.rollout import GenerationRequest, MockEngine, WeightSyncer, collate_samples
from kiln.rollout.types import Sample


def _mk_batch(version: int, n: int = 2):
    s = [
        Sample([1, 2], [3, 4], [-0.1, -0.2], group_id=0, policy_version=version, reward=1.0) for _ in range(n)
    ]
    return collate_samples(s)


# ------------------------------------------------------------------ buffer
def test_buffer_fifo_and_staleness_drop():
    buf = ExperienceBuffer(capacity=4, max_staleness=1)
    for v in range(4):
        assert buf.put(_mk_batch(v), v)
    # trainer is at version 3: versions 0,1 are too stale (3-0=3, 3-1=2 > 1), 2 and 3 are fine
    b = buf.get(current_version=3)
    assert int(b.policy_version[0]) == 2
    assert buf.stats.dropped_batches == 2 and buf.stats.dropped_samples == 4
    assert int(buf.get(3).policy_version[0]) == 3
    assert buf.get(3, timeout=0.05) is None
    m = buf.metrics()
    assert m["buffer/dropped_batches"] == 2 and m["buffer/mean_staleness"] == 0.5


def test_buffer_backpressure_blocks_put_until_get():
    buf = ExperienceBuffer(capacity=1, max_staleness=10)
    assert buf.put(_mk_batch(0), 0)
    assert not buf.put(_mk_batch(1), 1, timeout=0.05)  # full
    done = threading.Event()

    def producer():
        assert buf.put(_mk_batch(1), 1, timeout=5)
        done.set()

    threading.Thread(target=producer).start()
    time.sleep(0.05)
    assert not done.is_set()
    buf.get(1)
    assert done.wait(2)


def test_buffer_close_unblocks_waiters():
    buf = ExperienceBuffer(capacity=1, max_staleness=1)
    out = {}

    def consumer():
        out["v"] = buf.get(0, timeout=5)

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)
    buf.close()
    t.join(2)
    assert out["v"] is None and buf.closed


def test_buffer_resize_grows_capacity():
    buf = ExperienceBuffer(capacity=1, max_staleness=5)
    buf.put(_mk_batch(0), 0)
    assert not buf.put(_mk_batch(0), 0, timeout=0.01)
    buf.resize(2)
    assert buf.put(_mk_batch(0), 0, timeout=0.01)
    assert len(buf) == 2


# --------------------------------------------------------------- staleness
def test_staleness_controller_grows_when_training_is_bottleneck():
    c = StalenessController(depth=1, max_depth=4, max_staleness=4, cooldown=1)
    for _ in range(5):
        c.observe(rollout_seconds=1.0, train_seconds=3.5)
        c.step()
    assert c.depth == 4


def test_staleness_controller_shrinks_when_rollouts_are_bottleneck():
    c = StalenessController(depth=4, max_depth=4, max_staleness=4, cooldown=1)
    for _ in range(5):
        c.observe(rollout_seconds=5.0, train_seconds=1.0)
        c.step()
    assert c.depth == 1


def test_staleness_controller_respects_max_staleness_and_cooldown():
    c = StalenessController(depth=1, max_depth=8, max_staleness=2, cooldown=3)
    c.observe(rollout_seconds=1.0, train_seconds=10.0)
    assert c.step() == 1 and c.step() == 1  # cooldown
    assert c.step() == 2
    for _ in range(10):
        c.step()
    assert c.depth == 2  # capped by max_staleness


# ---------------------------------------------------------------- worker
def test_rollout_worker_produces_scored_batches():
    cfg = KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32"},
            "train": {"rollout_batch_size": 4, "micro_batch_size": 4},
            "rollout": {"sampling": {"n": 2, "max_new_tokens": 5}},
        }
    )
    tok = ByteTokenizer()
    model = TinyLM(TinyConfig(hidden=32, layers=1, heads=2))
    engine = MockEngine(model, eos_id=tok.eos_id)
    ds = PromptDataset(cfg.data, tok, rows=[{"prompt": "a", "answer": "1"}, {"prompt": "b", "answer": "2"}])
    rewards_seen = []

    def reward_fn(items):
        rewards_seen.append(len(items))
        return [float(len(it["response"])) for it in items], [
            {"len": float(len(it["response"]))} for it in items
        ]

    buf = ExperienceBuffer(2, 2)
    w = RolloutWorker(0, cfg, engine, ds.batches(2, seed=0), reward_fn, buf, tok, 2)
    batch, version = w.produce_batch()
    assert batch.batch_size == 4 and version == 0
    assert rewards_seen == [4]
    assert batch.rewards.shape == (4,)
    assert all(s.reward is not None and "len" in s.reward_breakdown for s in batch.samples)
    assert batch.response_mask.sum() > 0
    m = w.metrics()
    assert m["rollout/samples"] == 4 and "reward/len" in m


def test_rollout_worker_thread_fills_buffer_and_stops():
    cfg = KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32"},
            "train": {"rollout_batch_size": 2, "micro_batch_size": 2},
            "rollout": {"sampling": {"n": 1, "max_new_tokens": 3}},
        }
    )
    tok = ByteTokenizer()
    engine = MockEngine(TinyLM(TinyConfig(hidden=32, layers=1, heads=2)), eos_id=tok.eos_id)
    ds = PromptDataset(cfg.data, tok, rows=[{"prompt": "a"}])
    buf = ExperienceBuffer(2, 100)
    w = RolloutWorker(
        0,
        cfg,
        engine,
        ds.batches(2, seed=0),
        lambda items: ([0.0] * len(items), [{}] * len(items)),
        buf,
        tok,
        2,
    )
    w.start()
    deadline = time.time() + 5
    while len(buf) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert len(buf) == 2
    buf.close()
    w.stop()
    assert w.error is None


# ------------------------------------------------------------ weight sync
def test_weight_sync_copies_into_engine_and_tags_version():
    model = TinyLM(TinyConfig(hidden=32, layers=1, heads=2))
    eng = MockEngine(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    before = eng.model.embed.weight.clone()
    stats = WeightSyncer([eng], transport="inproc").sync(model, version=7)
    assert eng.policy_version == 7
    assert not torch.equal(before, eng.model.embed.weight)
    assert torch.equal(model.embed.weight, eng.model.embed.weight)
    assert stats.num_tensors > 0 and stats.bytes > 0 and stats.num_buckets >= 1
    # generations after sync carry the new version
    s = eng.generate(
        [GenerationRequest([1, 2, 3], group_id=0, n=1)],
        __import__("kiln.config.schema", fromlist=["SamplingConfig"]).SamplingConfig(max_new_tokens=2),
    )
    assert s[0].policy_version == 7


def test_weight_sync_filter_fn_reduces_payload():
    model = TinyLM(TinyConfig(hidden=32, layers=1, heads=2))
    eng = MockEngine(model)
    full = WeightSyncer([eng]).sync(model, 1)
    part = WeightSyncer([eng], filter_fn=lambda n: "embed" in n).sync(model, 2)
    assert part.num_tensors < full.num_tensors and part.bytes < full.bytes
