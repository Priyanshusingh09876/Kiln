"""Benchmarks: ``kiln bench kernels|step|scheduler|sync``.

Each benchmark measures on *this* machine and reports what it measured. On a
CPU-only box the kernel benchmark compares the PyTorch reference path to the
naive un-fused path (it still shows the memory-traffic win); on a CUDA box it
additionally times the Triton kernels and reports achieved bandwidth.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from kiln import native


def _timeit(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench_kernels(device: str | None = None) -> list[dict[str, Any]]:
    from kiln.kernels import apply_rope, fused_logprob_entropy, rmsnorm, rope_tables

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    rows = []
    shapes = [(2048, 32768), (4096, 128256)] if device == "cuda" else [(256, 4096), (512, 8192)]
    for n, v in shapes:
        logits = torch.randn(n, v, device=device, dtype=dtype, requires_grad=True)
        tgt = torch.randint(0, v, (n,), device=device)

        def naive(logits=logits, tgt=tgt):
            lp = torch.log_softmax(logits.float(), -1)
            tok = lp.gather(-1, tgt[:, None]).squeeze(-1)
            ent = -(lp.exp() * lp).sum(-1)
            (tok.sum() + ent.sum()).backward()

        def fused(logits=logits, tgt=tgt):
            lp, ent = fused_logprob_entropy(logits, tgt)
            (lp.sum() + ent.sum()).backward()

        t_naive, t_fused = _timeit(naive), _timeit(fused)
        bytes_min = n * v * logits.element_size() * 2  # read logits fwd + bwd, write grad
        rows.append(
            {
                "kernel": "logprob+entropy fwd+bwd",
                "shape": f"{n}x{v}",
                "backend": native.backend(),
                "naive_ms": t_naive * 1e3,
                "fused_ms": t_fused * 1e3,
                "speedup": t_naive / t_fused,
                "eff_GB_s": bytes_min / t_fused / 1e9,
            }
        )
    for n, h in [(8192, 4096), (16384, 8192)] if device == "cuda" else [(1024, 1024)]:
        x = torch.randn(n, h, device=device, dtype=dtype, requires_grad=True)
        w = torch.ones(h, device=device, dtype=dtype, requires_grad=True)

        def naive_rms(x=x, w=w):
            y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            y.sum().backward()

        def kiln_rms(x=x, w=w):
            rmsnorm(x, w).sum().backward()

        tn, tk = _timeit(naive_rms), _timeit(kiln_rms)
        rows.append(
            {
                "kernel": "rmsnorm fwd+bwd",
                "shape": f"{n}x{h}",
                "backend": native.backend(),
                "naive_ms": tn * 1e3,
                "fused_ms": tk * 1e3,
                "speedup": tn / tk,
                "eff_GB_s": n * h * x.element_size() * 3 / tk / 1e9,
            }
        )
    B, T, H, D = (4, 4096, 32, 128) if device == "cuda" else (2, 256, 8, 64)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    cos, sin = rope_tables(T, D, device=device)
    pos = torch.arange(T, device=device)[None].expand(B, T)

    def naive_rope():
        x1, x2 = q[..., : D // 2], q[..., D // 2 :]
        c, s = cos[pos][:, :, None, :], sin[pos][:, :, None, :]
        y = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)
        y.sum().backward()

    def kiln_rope():
        apply_rope(q, cos, sin, pos).sum().backward()

    tn, tk = _timeit(naive_rope), _timeit(kiln_rope)
    rows.append(
        {
            "kernel": "rope fwd+bwd",
            "shape": f"{B}x{T}x{H}x{D}",
            "backend": native.backend(),
            "naive_ms": tn * 1e3,
            "fused_ms": tk * 1e3,
            "speedup": tn / tk,
            "eff_GB_s": q.numel() * q.element_size() * 3 / tk / 1e9,
        }
    )
    return rows


def bench_step(config: Path | None) -> list[dict[str, Any]]:
    from kiln.config.schema import KilnConfig
    from kiln.orchestration import AsyncController

    cfg = (
        KilnConfig.from_yaml(config)
        if config
        else KilnConfig.model_validate(
            {
                "model": {"dtype": "fp32"},
                "train": {"rollout_batch_size": 16, "micro_batch_size": 8, "total_steps": 8},
                "rollout": {"sampling": {"n": 4, "max_new_tokens": 32}},
                "checkpoint": {"interval": 0},
                "logging": {"log_dir": "/tmp/kiln-bench", "run_name": "bench"},
            }
        )
    )
    rows = []
    for enabled in (False, True):
        cfg.async_.enabled = enabled
        rows_src = [{"prompt": f"{i}+{i}=", "answer": str(2 * i)} for i in range(64)]
        ctl = AsyncController.from_config(cfg, rows=rows_src)
        t0 = time.perf_counter()
        res = ctl.run()
        wall = time.perf_counter() - t0
        rows.append(
            {
                "mode": "async" if enabled else "sync",
                "steps": res.steps,
                "wall_s": wall,
                "steps_per_s": res.steps / wall,
                "mean_staleness": sum(res.series("async/mean_staleness")) / max(res.steps, 1),
                "mfu": sum(res.series("perf/mfu")) / max(res.steps, 1),
            }
        )
    return rows


def bench_scheduler() -> list[dict[str, Any]]:
    from kiln.serving.scheduler import SchedulerConfig, simulate

    rows = []
    for rps in (5, 10, 20, 40):
        for chunked in (False, True):
            r = simulate(
                SchedulerConfig(chunked_prefill=chunked, max_num_batched_tokens=2048),
                num_requests=1000,
                arrival_rps=rps,
                seed=0,
            )
            rows.append(
                {
                    "arrival_rps": rps,
                    "chunked_prefill": chunked,
                    "throughput_rps": r.throughput_rps,
                    "ttft_p50_ms": r.ttft_p50 * 1e3,
                    "ttft_p99_ms": r.ttft_p99 * 1e3,
                    "e2e_p99_s": r.e2e_p99,
                    "preemptions": r.preemptions,
                    "peak_kv_util": r.peak_kv_util,
                }
            )
    return rows


def bench_sync() -> list[dict[str, Any]]:
    from kiln.model import TinyConfig, TinyLM
    from kiln.rollout import MockEngine, WeightSyncer

    rows = []
    for hidden, layers in ((256, 4), (512, 8)):
        model = TinyLM(TinyConfig(hidden=hidden, layers=layers, heads=8, vocab_size=1024))
        engines = [MockEngine(model) for _ in range(2)]
        syncer = WeightSyncer(engines, transport="inproc")
        t = _timeit(lambda m=model, sy=syncer: sy.sync(m, 1), iters=5, warmup=1)
        s = syncer.history[-1]
        rows.append(
            {
                "params_M": model.num_params() / 1e6,
                "engines": 2,
                "bytes_MB": s.bytes / 1e6,
                "sync_ms": t * 1e3,
                "Gbps": s.bytes * 8 / t / 1e9,
            }
        )
    return rows


def run_benchmark(what: str, config: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    if what == "kernels":
        return {"kernels": bench_kernels()}
    if what == "step":
        return {"train_step": bench_step(config)}
    if what == "scheduler":
        return {"scheduler": bench_scheduler()}
    if what == "sync":
        return {"weight_sync": bench_sync()}
    if what == "all":
        return {
            **run_benchmark("kernels"),
            **run_benchmark("scheduler"),
            **run_benchmark("sync"),
            **run_benchmark("step", config),
        }
    raise ValueError(f"unknown benchmark {what!r}")
