from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("KILN_LOG_LEVEL", "WARNING")
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def tiny_cfg(tmp_path):
    from kiln.config.schema import KilnConfig

    return KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2},
            "train": {
                "rollout_batch_size": 8,
                "micro_batch_size": 4,
                "total_steps": 4,
                "optim": {"lr": 1e-3, "warmup_steps": 1},
            },
            "rollout": {"sampling": {"n": 4, "max_new_tokens": 8}},
            "async": {"queue_depth": 2, "max_staleness": 2},
            "checkpoint": {"dir": str(tmp_path / "ckpt"), "interval": 2},
            "logging": {"log_dir": str(tmp_path / "runs"), "run_name": "test"},
        }
    )


@pytest.fixture
def rows():
    return [{"prompt": f"{i}+{i}=", "answer": str(2 * i)} for i in range(32)]


def requires_cuda():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
