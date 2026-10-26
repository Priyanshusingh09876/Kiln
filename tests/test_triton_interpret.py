"""Executes the actual Triton kernels on CPU via Triton's interpreter.

Added after the fused log-prob forward returned +inf on an RTX 3080: a
formatting change had moved ``+ tl.sum(e)`` into a comment, so the softmax
denominator stayed 0. The GPU-only tests were skipped in CPU CI and nothing
else executed the kernel body.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
ROOT = Path(__file__).resolve().parents[1]


def test_fused_logprob_triton_kernel_in_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests/interp/check_fused_logprob.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
