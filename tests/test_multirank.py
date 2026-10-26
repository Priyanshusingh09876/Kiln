"""Two real processes, real gloo process groups, real FSDP2 sharding — on CPU.

Runs tests/multirank/run_dp2.py under torchrun and checks its RESULT line. This is
the test that exercises the distributed code paths (device mesh, FSDP2 gradient
reduction, full-state gather for weight sync, rank-sharded checkpoints, sharded
resume) that single-process tests cannot reach.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist

SCRIPT = Path(__file__).parent / "multirank" / "run_dp2.py"


@pytest.mark.slow
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="needs torch.distributed with gloo"
)
@pytest.mark.skipif(sys.platform == "win32", reason="torchrun + gloo on Windows is unreliable")
def test_two_rank_fsdp_training_and_exact_resume(tmp_path):
    env = {**os.environ, "KILN_MR_DIR": str(tmp_path), "OMP_NUM_THREADS": "1", "KILN_LOG_LEVEL": "WARNING"}
    torchrun = shutil.which("torchrun")
    cmd = ([torchrun] if torchrun else [sys.executable, "-m", "torch.distributed.run"]) + [
        "--nproc_per_node=2",
        "--standalone",
        str(SCRIPT),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, (
        f"no RESULT line (exit {proc.returncode})\nstdout:\n{proc.stdout[-3000:]}\nstderr:\n{proc.stderr[-3000:]}"
    )
    r = json.loads(lines[-1][len("RESULT ") :])
    assert r["sharded"], "parameters are not FSDP2 DTensors"
    assert r["disjoint_prompts"], "both ranks trained on the same prompts"
    assert r["ranks_in_sync"], "gathered weights differ across ranks: gradients not reduced"
    assert r["resume_steps"] == 4
    assert r["resume_weights_equal"], "resumed multi-rank run diverged from uninterrupted run"
    assert r["resume_losses_equal"]
    assert {"rank0.pt", "rank1.pt", "meta.json"} <= set(r["shard_files"])
    assert r["world_size"] == 2 and r["backend"] == "gloo"
    assert proc.returncode == 0
