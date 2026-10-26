"""Offline DPO: learns a preference, resumes exactly, and runs from the CLI."""

from __future__ import annotations

import json
import random

import torch

from kiln.config.schema import KilnConfig
from kiln.trainer.dpo import DPORunner


def _rows(n=48, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        p = "".join(rng.choice("abcdefgh") for _ in range(6))
        rows.append({"prompt": f"q:{p}", "chosen": "yes", "rejected": "no!"})
    return rows


def _cfg(tmp_path, ckpt="ck", total=30, run="dpo"):
    return KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2},
            "train": {
                "rollout_batch_size": 16,
                "micro_batch_size": 8,
                "total_steps": total,
                "optim": {"lr": 5e-3, "warmup_steps": 1},
            },
            "algorithm": {"name": "dpo", "dpo_beta": 0.1, "kl": {"coef": 0.0}},
            "async": {"enabled": False, "queue_depth": 1, "max_staleness": 0},
            "checkpoint": {"dir": str(tmp_path / ckpt), "interval": 10},
            "logging": {"log_dir": str(tmp_path / "runs"), "run_name": run},
        }
    )


def test_dpo_learns_preference(tmp_path):
    res = DPORunner.from_config(_cfg(tmp_path), rows=_rows()).run()
    acc = [m["dpo/accuracy"] for m in res.history]
    margin = [m["dpo/margin"] for m in res.history]
    loss = [m["loss/total"] for m in res.history]
    assert res.steps == 30
    assert abs(loss[0] - 0.6931) < 0.01  # log(2): policy == reference at step 0
    assert sum(acc[-5:]) / 5 == 1.0 and margin[-1] > 1.0 and loss[-1] < 0.3


def test_dpo_resume_is_exact(tmp_path):
    a = DPORunner.from_config(_cfg(tmp_path, "a", 20, "a"), rows=_rows())
    ra = a.run()
    b1 = DPORunner.from_config(_cfg(tmp_path, "b", 20, "b1"), rows=_rows())
    b1.run(max_steps=10)
    b2 = DPORunner.from_config(_cfg(tmp_path, "b", 20, "b2"), rows=_rows())
    rb = b2.run()
    assert [m["loss/total"] for m in ra.history[10:]] == [m["loss/total"] for m in rb.history]
    for k, v in a.trainer.model.state_dict().items():
        assert torch.equal(v, b2.trainer.model.state_dict()[k])


def test_dpo_from_cli(tmp_path):
    from typer.testing import CliRunner

    from kiln.cli import app

    data = tmp_path / "pref.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in _rows()), encoding="utf-8")
    y = tmp_path / "dpo.yaml"
    y.write_text(_cfg(tmp_path, total=3).to_yaml(), encoding="utf-8")
    r = CliRunner().invoke(app, ["train", str(y), "--set", f"data.train_path={data}"])
    assert r.exit_code == 0, r.output
    assert "3 DPO steps" in r.output
