"""Async checkpointer (atomicity, resume, pruning) and config schema."""

from __future__ import annotations

import os
import textwrap

import pytest
import torch

from kiln.checkpoint import AsyncCheckpointer
from kiln.config.schema import CheckpointConfig, KilnConfig


def _state(seed: int):
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(4, 4, generator=g), "b": torch.randn(4, generator=g)}


def test_save_load_roundtrip_async(tmp_path):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=True, keep_last=5))
    st = ck.save(
        1,
        _state(1),
        {"step": 1, "optimizer": {"lr": 0.1, "state": {0: {"m": torch.ones(2)}}, "groups": [{"lr": 0.1}]}},
    )
    assert st.copy_seconds >= 0
    ck.wait()
    assert ck.latest_step() == 1
    model, extra, meta = ck.load()
    assert torch.equal(model["w"], _state(1)["w"])
    assert extra["step"] == 1 and extra["optimizer"]["lr"] == 0.1
    assert torch.equal(extra["optimizer"]["state"][0]["m"], torch.ones(2)) and extra["optimizer"][
        "groups"
    ] == [{"lr": 0.1}]
    assert meta["step"] == 1


def test_save_is_atomic_no_tmp_left_and_latest_updated_last(tmp_path):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=False))
    ck.save(5, _state(5), {})
    assert (tmp_path / "step_5").is_dir()
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())
    assert (tmp_path / "LATEST").read_text().strip() == "5"


def test_training_can_mutate_source_tensors_while_write_in_flight(tmp_path):
    """The whole point: we copy to host buffers, so mutating the live weights after save() is safe."""
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=True))
    live = {"w": torch.zeros(1000, 1000)}
    ck.save(1, live, {})
    live["w"].fill_(1.0)  # simulate the next optimizer step
    ck.wait()
    model, _, _ = ck.load(1)
    assert model["w"].sum() == 0


def test_backpressure_one_write_in_flight(tmp_path):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=True, keep_last=10))
    big = {"w": torch.zeros(2000, 2000)}
    for s in range(1, 4):
        ck.save(s, big, {})
    ck.wait()
    assert ck.latest_step() == 3 and len(ck.history) == 3


def test_prune_keeps_last_n(tmp_path):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=False, keep_last=2))
    for s in (1, 2, 3, 4):
        ck.save(s, _state(s), {})
    dirs = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("step_"))
    assert dirs == ["step_3", "step_4"]


def test_writer_error_is_surfaced_on_wait(tmp_path, monkeypatch):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=True))

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ck, "_write", boom)
    ck.save(1, _state(1), {})
    with pytest.raises(RuntimeError, match="checkpoint write failed") as ei:
        ck.wait()
    assert isinstance(ei.value.__cause__, OSError)


def test_overhead_fraction(tmp_path):
    ck = AsyncCheckpointer(CheckpointConfig(dir=str(tmp_path), async_save=True))
    ck.save(10, _state(1), {})
    ck.wait()
    assert 0 <= ck.overhead_fraction(step_seconds=1.0) < 0.5


# ------------------------------------------------------------------ config
def test_config_defaults_and_derived():
    cfg = KilnConfig()
    assert cfg.algorithm.name == "grpo" and cfg.rollout.engine == "mock"
    assert cfg.parallel.world_size == 1
    assert cfg.algorithm.needs_reference and not cfg.algorithm.needs_critic
    assert KilnConfig.model_validate({"algorithm": {"name": "ppo"}}).algorithm.needs_critic


def test_config_strict_rejects_typos():
    with pytest.raises(Exception, match="extra"):
        KilnConfig.model_validate({"train": {"rollout_batch_sz": 8}})


def test_config_cross_field_validation():
    with pytest.raises(ValueError, match="divisible"):
        KilnConfig.model_validate({"train": {"rollout_batch_size": 10, "micro_batch_size": 4}})
    with pytest.raises(ValueError, match="sglang_urls"):
        KilnConfig.model_validate({"rollout": {"engine": "sglang"}})
    with pytest.raises(ValueError, match="offline"):
        KilnConfig.model_validate({"algorithm": {"name": "dpo"}})
    assert (
        KilnConfig.model_validate({"algorithm": {"name": "dpo"}, "async": {"enabled": False}}).algorithm.name
        == "dpo"
    )


def test_yaml_env_expansion_and_overrides(tmp_path, monkeypatch):
    y = tmp_path / "c.yaml"
    y.write_text(
        textwrap.dedent("""
        model: {name_or_path: "${KILN_MODEL:-tiny}", dtype: fp32}
        data: {train_path: "${DATA_DIR}/train.jsonl"}
        train: {total_steps: 5}
    """)
    )
    with pytest.raises(KeyError):
        KilnConfig.from_yaml(y)
    monkeypatch.setenv("DATA_DIR", "/data")
    cfg = KilnConfig.from_yaml(
        y, overrides=["train.total_steps=9", "rollout.sampling.n=16", "async.enabled=false"]
    )
    assert cfg.data.train_path == "/data/train.jsonl" and cfg.model.name_or_path == "tiny"
    assert cfg.train.total_steps == 9 and cfg.rollout.sampling.n == 16 and cfg.async_.enabled is False
    again = KilnConfig.model_validate(__import__("yaml").safe_load(cfg.to_yaml()))
    assert again == cfg


def test_all_shipped_configs_validate():
    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    files = [f for f in os.listdir(root) if f.endswith(".yaml")]
    assert files, "no configs shipped"
    for f in files:
        KilnConfig.from_yaml(os.path.join(root, f))
