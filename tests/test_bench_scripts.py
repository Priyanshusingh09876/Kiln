"""Smoke tests for the benchmark scripts, so they cannot rot before someone runs them on a GPU."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kernel_bench_cases_run_and_pass_on_cpu():
    kb = _load("bench_kernels")
    av = kb.available_backends("cpu")
    assert av["naive"] is None and av["torch"] is None and av["triton"] and av["cuda"]
    for be in ("naive", "torch"):
        r = kb.case_logprob(16, 300, torch.float32, "cpu", be, iters=1)
        assert max(r["err"].values()) < 1e-4 and r["peak_mb"] is None
        r = kb.case_rmsnorm(8, 64, torch.float32, "cpu", be, iters=1)
        assert max(r["err"].values()) < 1e-4
        r = kb.case_rope(1, 16, 2, 8, torch.float32, "cpu", be, iters=1)
        assert max(r["err"].values()) < 1e-5


@pytest.mark.slow
def test_sync_vs_async_harness_runs(tmp_path):
    sva = _load("bench_sync_vs_async")
    base = sva.cpu_sim_config(steps=3, engines=2, ms_per_token=0.0)
    base["logging"]["log_dir"] = str(tmp_path)
    rows = [{"prompt": f"p{i}"} for i in range(16)]
    for mode in ("sync", "async-d2"):
        r = sva.run_mode(base, mode, seed=0, rows=rows, reward_fn=sva.first_byte_reward)
        assert r["steps"] == 3 and 0 < r["trainer_busy_frac"] <= 1
        assert r["gpu_util_mean"] is None  # no GPU here: reported as n/a, never invented
        assert len(r["reward_curve"]) == 3
