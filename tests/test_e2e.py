"""End-to-end: the full async GRPO loop on CPU with the tiny model.

The learnable task: reward ``1 - |first_byte - '7'| / 128`` (0 if the first
token is not a byte at all). A random 1-layer model scores ~0.25 on average;
a correct GRPO implementation drives the first byte toward ``'7'`` within a
few dozen steps. If any piece of the alignment chain (collation offset,
packing/unpacking, log-prob kernel, advantages, clipping) is wrong, this
test fails to learn.
"""

from __future__ import annotations

import json

import pytest
import torch

from kiln.config.schema import KilnConfig
from kiln.orchestration import AsyncController
from kiln.rollout.types import Sample
from kiln.utils.logging import read_metrics

TARGET = ord("7")


def first_byte_reward(items):
    r = []
    for it in items:
        c = it["response"][:1]
        r.append(max(0.0, 1.0 - abs(ord(c) - TARGET) / 128) if c else 0.0)
    return r, [{"first7": v} for v in r]


def _cfg(tmp_path, **over):
    base = {
        "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2, "tiny_vocab": 512},
        "train": {
            "rollout_batch_size": 32,
            "micro_batch_size": 16,
            "total_steps": 60,
            "pack_sequences": True,
            "optim": {"lr": 1e-2, "warmup_steps": 2, "grad_clip": 1.0},
        },
        "rollout": {"num_engines": 2, "sampling": {"n": 8, "max_new_tokens": 4, "temperature": 1.0}},
        "algorithm": {"name": "grpo", "kl": {"coef": 0.0}},
        "async": {"enabled": True, "queue_depth": 2, "max_staleness": 2, "is_correction": "token"},
        "checkpoint": {"dir": str(tmp_path / "ckpt"), "interval": 20},
        "logging": {"log_dir": str(tmp_path / "runs"), "run_name": "e2e"},
    }
    for k, v in over.items():
        base.setdefault(k, {}).update(v)
    return KilnConfig.model_validate(base)


ROWS = [{"prompt": f"p{i}"} for i in range(16)]


@pytest.mark.parametrize("algo", ["grpo", "dr_grpo", "rloo"])
def test_async_loop_learns_first_byte_task(tmp_path, algo):
    cfg = _cfg(tmp_path, algorithm={"name": algo})
    ctl = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward)
    res = ctl.run()
    assert res.steps == 60
    rewards = res.series("reward/mean")
    early, late = sum(rewards[:5]) / 5, sum(rewards[-5:]) / 5
    assert late > 0.85, f"{algo}: reward did not improve ({early:.3f} -> {late:.3f})"
    assert late > early + 0.3
    # async bookkeeping is real: some batches were stale, nothing exceeded max_staleness
    stale = res.series("async/mean_staleness")
    assert max(stale) <= cfg.async_.max_staleness and sum(stale) > 0
    # metrics landed on disk
    recs = read_metrics(res.run_dir / "metrics.jsonl")
    assert len(recs) == 60 and "perf/mfu" in recs[-1]


def test_sync_mode_is_deterministic_and_on_policy(tmp_path):
    cfg = _cfg(
        tmp_path,
        **{"async": {"enabled": False, "queue_depth": 1, "max_staleness": 0}, "train": {"total_steps": 6}},
    )
    cfg.rollout.num_engines = 1
    a = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward).run()
    cfg.logging.run_name = "e2e2"
    cfg.checkpoint.dir = str(tmp_path / "ckpt2")
    b = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward).run()
    assert a.series("loss/total") == b.series("loss/total")
    assert all(s == 0 for s in a.series("async/mean_staleness"))
    # on-policy: the engine's behaviour log-probs equal the trainer's -> importance weights are exactly 1
    assert all(abs(w - 1.0) < 1e-4 for w in a.series("async/is_weight"))


def test_importance_weights_depart_from_one_when_stale(tmp_path):
    cfg = _cfg(tmp_path, **{"async": {"queue_depth": 3, "max_staleness": 3}, "train": {"total_steps": 12}})
    res = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward).run()
    w = res.series("async/is_weight")
    assert any(abs(x - 1.0) > 1e-3 for x in w[3:]), "stale batches must produce non-trivial IS weights"
    assert max(w) <= cfg.async_.is_clip + 1e-6


def test_checkpoint_resume_continues_from_same_state(tmp_path):
    cfg = _cfg(tmp_path, train={"total_steps": 10}, checkpoint={"interval": 5, "dir": str(tmp_path / "ck")})
    cfg.rollout.num_engines = 1
    cfg.async_.enabled = False
    ctl = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward)
    ctl.run(max_steps=5)
    assert ctl.ckpt.latest_step() == 5
    w5 = ctl.trainer.model.embed.weight.detach().clone()
    # new controller, fresh random init, must resume to the saved weights and step counter
    ctl2 = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward)
    assert not torch.equal(ctl2.trainer.model.embed.weight, w5)
    ctl2.maybe_resume()
    assert ctl2.trainer.step == 5 and ctl2.trainer.policy_version == 5
    assert torch.equal(ctl2.trainer.model.embed.weight, w5)
    assert "exp_avg" in next(iter(ctl2.trainer.optimizer.state.values()))
    res = ctl2.run()
    assert res.steps == 10
    meta = json.loads((tmp_path / "ck" / "step_10" / "meta.json").read_text())
    assert meta["step"] == 10


def test_ppo_with_critic_runs(tmp_path):
    cfg = _cfg(
        tmp_path,
        algorithm={"name": "ppo", "kl": {"coef": 0.01, "in_reward": True}},
        train={"total_steps": 4, "pack_sequences": False},
    )
    res = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward).run()
    assert res.steps == 4 and all(torch.isfinite(torch.tensor(v)) for v in res.series("loss/value"))


def test_kl_to_reference_is_zero_at_start_and_grows(tmp_path):
    cfg = _cfg(tmp_path, algorithm={"kl": {"coef": 0.05, "estimator": "k3"}}, train={"total_steps": 8})
    res = AsyncController.from_config(cfg, rows=ROWS, reward_fn=first_byte_reward).run()
    kl = res.series("kl/ref")
    assert kl[0] < 1e-6 and kl[-1] > kl[0]


def test_worker_failure_propagates(tmp_path):
    cfg = _cfg(tmp_path, train={"total_steps": 5})

    def bad_reward(items):
        raise RuntimeError("reward service down")

    ctl = AsyncController.from_config(cfg, rows=ROWS, reward_fn=bad_reward)
    with pytest.raises(RuntimeError, match="rollout worker"):
        ctl.run()


def test_cli_train_smoke(tmp_path):
    from typer.testing import CliRunner

    from kiln.cli import app

    y = tmp_path / "c.yaml"
    y.write_text(
        _cfg(tmp_path, train={"total_steps": 2}, reward={"functions": [{"name": "exact_match"}]}).to_yaml()
    )
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    r = CliRunner().invoke(app, ["train", str(y), "--set", f"data.train_path={tmp_path / 'train.jsonl'}"])
    assert r.exit_code == 0, r.output
    assert "done" in r.output
    r = CliRunner().invoke(app, ["validate", str(y)])
    assert r.exit_code == 0 and "OK" in r.output
    r = CliRunner().invoke(app, ["inspect-ckpt", str(tmp_path / "ckpt")])
    assert r.exit_code == 0 and "step_2" in r.output


def test_sample_dataclass_defaults():
    s = Sample([1], [2], [-0.1], 0, 0)
    assert s.total_len == 2 and s.reward is None and s.finish_reason == "stop"
