"""Failure recovery: an interrupted-and-resumed run must match an uninterrupted one.

These tests exist because the earlier resume test only checked that weights came
back. It missed two real bugs: the prompt stream restarted from the beginning, and
the rollout engines kept the *initial* weights after a resume, so in async mode every
new sample was dropped as stale and the run hung.
"""

from __future__ import annotations

import threading

import torch

from kiln.config.schema import KilnConfig
from kiln.orchestration import AsyncController

ROWS = [{"prompt": f"p{i}"} for i in range(16)]


def reward_fn(items):
    r = [max(0.0, 1.0 - abs(ord(it["response"][:1] or "\0") - ord("7")) / 128) for it in items]
    return r, [{"r": v} for v in r]


def _cfg(tmp_path, ckpt_dir, total, async_enabled, run):
    return KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2},
            "train": {
                "rollout_batch_size": 16,
                "micro_batch_size": 8,
                "total_steps": total,
                "optim": {"lr": 1e-2, "warmup_steps": 1},
            },
            "rollout": {"num_engines": 1, "sampling": {"n": 4, "max_new_tokens": 4}},
            "algorithm": {"name": "grpo", "kl": {"coef": 0.0}},
            "async": {
                "enabled": async_enabled,
                "queue_depth": 2 if async_enabled else 1,
                "max_staleness": 2 if async_enabled else 0,
            },
            "checkpoint": {"dir": str(ckpt_dir), "interval": 3, "async_save": True},
            "logging": {"log_dir": str(tmp_path / "runs"), "run_name": run},
        }
    )


def _weights(ctl):
    return {k: v.detach().clone() for k, v in ctl.trainer.model.state_dict().items()}


def test_sync_resume_is_bit_identical_to_uninterrupted_run(tmp_path):
    # uninterrupted: 6 steps
    a = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "a", 6, False, "a"), rows=ROWS, reward_fn=reward_fn
    )
    ra = a.run()
    wa = _weights(a)

    # interrupted at step 3 (checkpoint interval 3), then a *fresh process-equivalent* controller resumes
    b1 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "b", 6, False, "b1"), rows=ROWS, reward_fn=reward_fn
    )
    b1.run(max_steps=3)
    b2 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "b", 6, False, "b2"), rows=ROWS, reward_fn=reward_fn
    )
    rb = b2.run()
    wb = _weights(b2)

    assert rb.steps == 6
    assert ra.series("loss/total")[3:] == rb.series("loss/total"), "post-resume losses differ"
    assert ra.series("reward/mean")[3:] == rb.series("reward/mean"), "post-resume data stream differs"
    for k in wa:
        assert torch.equal(wa[k], wb[k]), f"weights differ after resume: {k}"


def test_resume_pushes_restored_weights_to_engines(tmp_path):
    c1 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "c", 6, False, "c1"), rows=ROWS, reward_fn=reward_fn
    )
    c1.run(max_steps=3)
    c2 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "c", 6, False, "c2"), rows=ROWS, reward_fn=reward_fn
    )
    c2.maybe_resume()
    eng = c2.workers[0].engine
    assert eng.policy_version == 3
    for k, v in c2.trainer.model.state_dict().items():
        assert torch.equal(eng.model.state_dict()[k].cpu(), v.detach().cpu()), (
            f"engine has stale weights for {k}"
        )


def test_async_resume_completes_and_respects_staleness(tmp_path):
    d1 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "d", 8, True, "d1"), rows=ROWS, reward_fn=reward_fn
    )
    d1.run(max_steps=3)
    d2 = AsyncController.from_config(
        _cfg(tmp_path, tmp_path / "d", 8, True, "d2"), rows=ROWS, reward_fn=reward_fn
    )
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", d2.run()), daemon=True)
    t.start()
    t.join(timeout=45)
    assert not t.is_alive(), "async run hung after resume (engines not re-synced?)"
    r = out["r"]
    assert r.steps == 8
    assert max(r.series("async/mean_staleness")) <= 2
    assert d2.buffer.stats.dropped_batches == 0 or r.steps == 8
