#!/usr/bin/env python
"""Kernel comparison: PyTorch reference vs Triton vs CUDA C++, with error and memory.

    python scripts/bench_kernels.py --tag cpu          # reference path only; correctness + timing
    python scripts/bench_kernels.py --tag h100         # on a GPU: every backend that is available

For every (kernel, shape, dtype, backend) it measures, forward + backward together:

* time — median over iterations, CUDA-synchronised;
* max absolute error of each output and of the input gradient against a float64
  reference, and a pass/fail against a per-dtype tolerance;
* peak extra device memory during fwd+bwd (``torch.cuda.max_memory_allocated``
  minus what was allocated before the call) — the number the fused kernels exist
  to reduce. ``n/a`` on CPU, where PyTorch does not expose it.

Backends are pinned with ``KILN_KERNEL_BACKEND``; a backend that is not available
on this machine is listed as skipped, never silently replaced by another. Writes
``docs/benchmarks/kernels_<tag>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from kiln import native
from kiln.kernels import apply_rope, fused_logprob_entropy, rmsnorm, rope_tables
from kiln.kernels.dispatch import triton_enabled

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
TOL = {torch.float32: 1e-4, torch.bfloat16: 3e-2, torch.float16: 1e-2}
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def available_backends(device: str) -> dict[str, str | None]:
    """backend -> reason it is unavailable (None if available)."""
    out: dict[str, str | None] = {"naive": None, "torch": None}
    if device != "cuda":
        out["triton"] = "no CUDA device"
        out["cuda"] = "no CUDA device"
        return out
    os.environ["KILN_KERNEL_BACKEND"] = "triton"
    out["triton"] = None if triton_enabled() else "triton not importable"
    out["cuda"] = None if native.cuda() is not None else "kiln._cuda not built (needs nvcc at install time)"
    os.environ["KILN_KERNEL_BACKEND"] = "auto"
    return out


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn: Callable[[], None], device: str, iters: int, warmup: int = 3) -> tuple[float, float | None]:
    for _ in range(warmup):
        fn()
    _sync(device)
    peak = None
    if device == "cuda":
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        fn()
        _sync(device)
        peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3, peak


# ------------------------------------------------------------- kernels
def case_logprob(n: int, v: int, dtype: torch.dtype, device: str, backend: str, iters: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(0)
    logits0 = (torch.randn(n, v, generator=g) * 3).to(device=device, dtype=dtype)
    tgt = torch.randint(0, v, (n,), generator=g).to(device)
    g1 = torch.randn(n, generator=g).to(device)
    g2 = torch.randn(n, generator=g).to(device)

    # float64 reference (computed once, on CPU to keep device memory clean)
    x64 = logits0.detach().cpu().double().requires_grad_(True)
    lp64 = torch.log_softmax(x64, -1)
    ref_lp = lp64.gather(-1, tgt.cpu()[:, None]).squeeze(-1)
    ref_ent = -(lp64.exp() * lp64).sum(-1)
    (ref_lp * g1.cpu().double() + ref_ent * g2.cpu().double()).sum().backward()
    ref_grad = x64.grad

    def op(x):
        if backend == "naive":
            lp = torch.log_softmax(x.float(), -1)
            return lp.gather(-1, tgt[:, None]).squeeze(-1), -(lp.exp() * lp).sum(-1)
        return fused_logprob_entropy(x, tgt)

    x = logits0.clone().requires_grad_(True)
    lp, ent = op(x)
    (lp * g1 + ent * g2).sum().backward()
    err = {
        "logp": (lp.detach().cpu().double() - ref_lp.detach()).abs().max().item(),
        "entropy": (ent.detach().cpu().double() - ref_ent.detach()).abs().max().item(),
        "grad": (x.grad.cpu().double() - ref_grad).abs().max().item(),
    }

    def fwd_bwd():
        xx = logits0.detach().requires_grad_(True)
        a, b = op(xx)
        (a.sum() + b.sum()).backward()

    ms, peak = measure(fwd_bwd, device, iters)
    return {
        "shape": f"{n}×{v}",
        "ms": ms,
        "peak_mb": peak,
        "err": err,
        "logits_mb": logits0.numel() * logits0.element_size() / 2**20,
    }


def case_rmsnorm(n: int, h: int, dtype: torch.dtype, device: str, backend: str, iters: int) -> dict:
    g = torch.Generator(device="cpu").manual_seed(1)
    x0 = torch.randn(n, h, generator=g).to(device=device, dtype=dtype)
    w0 = (1 + 0.1 * torch.randn(h, generator=g)).to(device=device, dtype=dtype)
    dy = torch.randn(n, h, generator=g).to(device=device, dtype=dtype)
    x64 = x0.cpu().double().requires_grad_(True)
    w64 = w0.cpu().double().requires_grad_(True)
    y64 = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + 1e-6) * w64
    (y64 * dy.cpu().double()).sum().backward()

    def op(x, w):
        if backend == "naive":
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
        return rmsnorm(x, w, 1e-6)

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    y = op(x, w)
    (y * dy).sum().backward()
    err = {
        "y": (y.detach().cpu().double() - y64.detach()).abs().max().item(),
        "dx": (x.grad.cpu().double() - x64.grad).abs().max().item(),
        "dw": (w.grad.cpu().double() - w64.grad).abs().max().item() / n,  # dw sums over rows; normalise
    }

    def fwd_bwd():
        xx = x0.detach().requires_grad_(True)
        ww = w0.detach().requires_grad_(True)
        op(xx, ww).sum().backward()

    ms, peak = measure(fwd_bwd, device, iters)
    return {"shape": f"{n}×{h}", "ms": ms, "peak_mb": peak, "err": err}


def case_rope(
    b: int, t: int, h: int, d: int, dtype: torch.dtype, device: str, backend: str, iters: int
) -> dict:
    g = torch.Generator(device="cpu").manual_seed(2)
    q0 = torch.randn(b, t, h, d, generator=g).to(device=device, dtype=dtype)
    cos, sin = rope_tables(t, d, device=device)
    pos = torch.arange(t, device=device)[None].expand(b, t)
    dy = torch.randn(b, t, h, d, generator=g).to(device=device, dtype=dtype)

    def naive(q, c, s):
        x1, x2 = q[..., : d // 2], q[..., d // 2 :]
        cc, ss = c[pos][:, :, None, :], s[pos][:, :, None, :]
        return torch.cat([x1 * cc - x2 * ss, x1 * ss + x2 * cc], -1)

    q64 = q0.cpu().double().requires_grad_(True)
    pos_cpu = pos.cpu()
    x1, x2 = q64[..., : d // 2], q64[..., d // 2 :]
    cc, ss = cos.cpu().double()[pos_cpu][:, :, None, :], sin.cpu().double()[pos_cpu][:, :, None, :]
    y64 = torch.cat([x1 * cc - x2 * ss, x1 * ss + x2 * cc], -1)
    (y64 * dy.cpu().double()).sum().backward()

    def op(q):
        return naive(q, cos, sin) if backend == "naive" else apply_rope(q, cos, sin, pos)

    q = q0.clone().requires_grad_(True)
    y = op(q)
    (y * dy).sum().backward()
    err = {
        "y": (y.detach().cpu().double() - y64.detach()).abs().max().item(),
        "dq": (q.grad.cpu().double() - q64.grad).abs().max().item(),
    }

    def fwd_bwd():
        qq = q0.detach().requires_grad_(True)
        op(qq).sum().backward()

    ms, peak = measure(fwd_bwd, device, iters)
    return {"shape": f"{b}×{t}×{h}×{d}", "ms": ms, "peak_mb": peak, "err": err}


# ---------------------------------------------------------------- driver
def plan(device: str, dtypes: list[str]) -> list[tuple[str, Callable, tuple, set[str]]]:
    """(kernel, fn, shape-args, backends that implement it)."""
    if device == "cuda":
        lp = [(4096, 32000), (4096, 128256), (2048, 151936)]
        rn = [(16384, 4096), (8192, 8192)]
        rp = [(4, 4096, 32, 128)]
    else:
        lp = [(256, 4096), (256, 32000)]
        rn = [(2048, 1024)]
        rp = [(2, 256, 8, 64)]
    out = []
    for dt in dtypes:
        out += [
            ("logprob+entropy", case_logprob, (n, v, DT[dt]), {"naive", "torch", "triton", "cuda"})
            for n, v in lp
        ]
        out += [
            ("rmsnorm", case_rmsnorm, (n, h, DT[dt]), {"naive", "torch", "triton", "cuda"}) for n, h in rn
        ]
        out += [("rope", case_rope, (*s, DT[dt]), {"naive", "torch", "triton"}) for s in rp]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtypes", nargs="+", default=None, help="default: fp32 on CPU, bf16 fp32 on GPU")
    ap.add_argument("--iters", type=int, default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.tag or (torch.cuda.get_device_name(0).replace(" ", "-").lower() if device == "cuda" else "cpu")
    dtypes = a.dtypes or (["bf16", "fp32"] if device == "cuda" else ["fp32"])
    iters = a.iters or (20 if device == "cuda" else 5)
    backends = available_backends(device)

    rows = []
    for kernel, fn, args, impls in plan(device, dtypes):
        for be in ("naive", "torch", "triton", "cuda"):
            if be not in impls:
                continue
            if backends[be] is not None:
                rows.append(
                    {
                        "kernel": kernel,
                        "dtype": str(args[-1]).replace("torch.", ""),
                        "backend": be,
                        "skipped": backends[be],
                    }
                )
                continue
            os.environ["KILN_KERNEL_BACKEND"] = {"naive": "torch", "torch": "torch"}.get(be, be)
            print(f"{kernel} {args} {be}", file=sys.stderr, flush=True)
            try:
                r = fn(*args, device=device, backend=be, iters=iters)
            except Exception as e:  # report, do not hide
                r = {"error": f"{type(e).__name__}: {e}"}
            r.update({"kernel": kernel, "dtype": str(args[-1]).replace("torch.", ""), "backend": be})
            if "err" in r:
                tol = TOL[args[-1]]
                r["tolerance"] = tol
                r["pass"] = all(
                    v <= tol * (10 if k in ("grad", "dx", "dw", "dq") else 1) for k, v in r["err"].items()
                )
            rows.append(r)
    os.environ["KILN_KERNEL_BACKEND"] = "auto"

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "device": torch.cuda.get_device_name(0)
        if device == "cuda"
        else f"CPU ({platform.processor() or platform.machine()})",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "backends": backends,
        "iters": iters,
        "command": " ".join(["python", "scripts/bench_kernels.py", *sys.argv[1:]]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"kernels_{tag}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    lines = [
        f"# Kernel comparison — `{tag}`",
        "",
        f"Generated {meta['date']} by `{meta['command']}` on **{meta['device']}** (torch {meta['torch']}, CUDA {meta['cuda']}).",
        f"Forward + backward, median of {iters} iterations. Errors are max |Δ| against a float64 reference;",
        "gradients are held to 10× the output tolerance. *naive* is the unfused PyTorch formula; *torch* is Kiln's",
        f"reference path; *triton* and *cuda* are the fused kernels. Raw data: `kernels_{tag}.json`.",
        "",
    ]
    skipped = {b: why for b, why in backends.items() if why}
    if skipped:
        lines += [
            "Not run on this machine: " + "; ".join(f"**{b}** ({why})" for b, why in skipped.items()) + ".",
            "",
        ]
    lines += [
        "| kernel | shape | dtype | backend | time | peak extra memory | max error (outputs / grad) | within tolerance |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "skipped" in r:
            continue
        if "error" in r:
            lines.append(f"| {r['kernel']} | | {r['dtype']} | {r['backend']} | | | **{r['error']}** | ❌ |")
            continue
        errs = " / ".join(f"{v:.1e}" for v in r["err"].values())
        mem = "n/a" if r["peak_mb"] is None else f"{r['peak_mb']:.0f} MB"
        lines.append(
            f"| {r['kernel']} | {r['shape']} | {r['dtype']} | {r['backend']} | {r['ms']:.2f} ms | {mem} | {errs} | "
            f"{'✅' if r['pass'] else '❌'} |"
        )
    if device == "cpu":
        lines += [
            "",
            "On CPU only the reference paths run, so this table establishes **correctness** of the reference",
            "implementations and the harness itself; timings are CPU timings and say nothing about the GPU kernels.",
            "Run the same command on a CUDA machine to fill in Triton and CUDA C++ rows, memory, and speed.",
        ]
    md = OUT / f"kernels_{tag}.md"
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    return (
        0
        if all(r.get("pass", True) for r in rows if "err" in r) and not any("error" in r for r in rows)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
