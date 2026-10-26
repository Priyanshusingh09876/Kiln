#!/usr/bin/env python
"""Kernel comparison: naive PyTorch, torch.compile, Liger Kernel, Kiln's Triton / CUDA C++ kernels.

    python scripts/bench_kernels.py --tag cpu          # reference path only; correctness + timing
    python scripts/bench_kernels.py --tag h100         # on a GPU: every backend that is available

For every (kernel, shape, dtype, backend) it measures, forward + backward together:

* time — median over iterations, CUDA-synchronised;
* max absolute error of each output and of the input gradient against a float64
  reference, and a pass/fail against a per-dtype tolerance;
* achieved memory bandwidth: the minimum bytes the operation must move (see
  ``min_bytes``) divided by the time, and that as a percentage of this GPU's
  measured device-to-device copy bandwidth — the practical roofline for
  memory-bound kernels like these;
* peak extra device memory during fwd+bwd (``torch.cuda.max_memory_allocated``
  minus what was allocated before the call) — the number the fused kernels exist
  to reduce. ``n/a`` on CPU, where PyTorch does not expose it.

Backends are pinned with ``KILN_KERNEL_BACKEND``; a backend that is not available
on this machine is listed as skipped, never silently replaced by another.
``compile`` is ``torch.compile`` of the naive formula. ``liger`` is Liger Kernel
(``pip install liger-kernel``; on Windows add ``--no-deps``): for log-prob it is
``LigerCrossEntropyLoss``, which computes cross-entropy only (no entropy) and
writes the gradient over its input during the forward pass. Writes
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
        for b in ("compile", "liger", "triton", "cuda"):
            out[b] = "no CUDA device"
        return out
    out["compile"] = None if hasattr(torch, "compile") else "torch.compile unavailable"
    try:
        import liger_kernel.transformers  # noqa: F401

        out["liger"] = None
    except Exception as e:  # noqa: BLE001
        out["liger"] = f"liger-kernel not importable ({type(e).__name__})"
    os.environ["KILN_KERNEL_BACKEND"] = "triton"
    out["triton"] = None if triton_enabled() else "triton not importable"
    out["cuda"] = None if native.cuda() is not None else "kiln._cuda not built (needs nvcc at install time)"
    os.environ["KILN_KERNEL_BACKEND"] = "auto"
    return out


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def copy_bandwidth_gbs(device: str) -> float | None:
    """Measured device-to-device copy bandwidth (GB/s): read + write of a 512 MB buffer."""
    if device != "cuda":
        return None
    n = 512 * 2**20
    a = torch.empty(n, dtype=torch.uint8, device=device)
    b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        b.copy_(a)
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n / best / 1e9


def theoretical_bandwidth_gbs() -> float | None:
    """Spec-sheet DRAM bandwidth from NVML (max memory clock x bus width), if pynvml is installed."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
        bits = pynvml.nvmlDeviceGetMemoryBusWidth(h)
        return 2 * mhz * 1e6 * bits / 8 / 1e9
    except Exception:  # noqa: BLE001
        return None


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

    def naive(x):
        lp = torch.log_softmax(x.float(), -1)
        return lp.gather(-1, tgt[:, None]).squeeze(-1), -(lp.exp() * lp).sum(-1)

    min_bytes = 3 * logits0.numel() * logits0.element_size()  # read (fwd) + read (bwd) + write grad

    if backend == "liger":
        from liger_kernel.transformers import LigerCrossEntropyLoss

        ce_none = LigerCrossEntropyLoss(reduction="none")
        ce_sum = LigerCrossEntropyLoss(reduction="sum")
        # its reference: d(sum CE)/dx in float64
        x64 = logits0.detach().cpu().double().requires_grad_(True)
        (-torch.log_softmax(x64, -1).gather(-1, tgt.cpu()[:, None])).sum().backward()
        x = logits0.clone().requires_grad_(True)
        loss = ce_none(x, tgt)
        loss.sum().backward()
        err = {
            "logp": (-loss.detach().float().cpu().double() - ref_lp.detach()).abs().max().item(),
            "grad": (x.grad.cpu().double() - x64.grad).abs().max().item(),
        }
        buf = logits0.clone()  # Liger overwrites its input with the gradient; timing is data-independent

        def fwd_bwd():
            ce_sum(buf.detach().requires_grad_(True), tgt).backward()

        ms, peak = measure(fwd_bwd, device, iters)
        return {
            "shape": f"{n}×{v}",
            "ms": ms,
            "peak_mb": peak,
            "err": err,
            "min_bytes": min_bytes,
            "note": "CE only, no entropy; gradient written in place",
        }

    if backend == "naive":
        op = naive
    elif backend == "compile":
        op = torch.compile(naive, dynamic=False)
    else:
        op = lambda x: fused_logprob_entropy(x, tgt)  # noqa: E731

    x = logits0.clone().requires_grad_(True)
    lp, ent = op(x)
    torch.autograd.backward([lp, ent], [g1, g2])
    err = {
        "logp": (lp.detach().cpu().double() - ref_lp.detach()).abs().max().item(),
        "entropy": (ent.detach().cpu().double() - ref_ent.detach()).abs().max().item(),
        "grad": (x.grad.cpu().double() - ref_grad).abs().max().item(),
    }

    def fwd_bwd():
        xx = logits0.detach().requires_grad_(True)
        a, b = op(xx)
        torch.autograd.backward([a, b], [g1, g2])

    ms, peak = measure(fwd_bwd, device, iters)
    return {
        "shape": f"{n}×{v}",
        "ms": ms,
        "peak_mb": peak,
        "err": err,
        "min_bytes": min_bytes,
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

    def naive(x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)

    if backend == "naive":
        op = naive
    elif backend == "compile":
        op = torch.compile(naive, dynamic=False)
    elif backend == "liger":
        from liger_kernel.transformers import LigerRMSNorm

        mod = LigerRMSNorm(h, eps=1e-6, in_place=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            mod.weight.copy_(w0)

        def op(x, w):  # the module owns its weight; dw is read from mod.weight.grad
            return mod(x)
    else:
        op = lambda x, w: rmsnorm(x, w, 1e-6)  # noqa: E731

    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    y = op(x, w)
    y.backward(dy)
    wgrad = mod.weight.grad if backend == "liger" else w.grad
    err = {
        "y": (y.detach().cpu().double() - y64.detach()).abs().max().item(),
        "dx": (x.grad.cpu().double() - x64.grad).abs().max().item(),
        "dw": (wgrad.cpu().double() - w64.grad).abs().max().item() / n,  # dw sums over rows; normalise
    }

    def fwd_bwd():
        xx = x0.detach().requires_grad_(True)
        ww = w0.detach().requires_grad_(True)
        op(xx, ww).backward(dy)

    ms, peak = measure(fwd_bwd, device, iters)
    # fwd: read x, write y; bwd: read x, read dy, write dx (weight and dw are negligible)
    min_bytes = 5 * x0.numel() * x0.element_size()
    return {"shape": f"{n}×{h}", "ms": ms, "peak_mb": peak, "err": err, "min_bytes": min_bytes}


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

    if backend == "naive":
        op = lambda q: naive(q, cos, sin)  # noqa: E731
    elif backend == "compile":
        cnaive = torch.compile(naive, dynamic=False)
        op = lambda q: cnaive(q, cos, sin)  # noqa: E731
    else:
        op = lambda q: apply_rope(q, cos, sin, pos)  # noqa: E731

    q = q0.clone().requires_grad_(True)
    y = op(q)
    y.backward(dy)
    err = {
        "y": (y.detach().cpu().double() - y64.detach()).abs().max().item(),
        "dq": (q.grad.cpu().double() - q64.grad).abs().max().item(),
    }

    def fwd_bwd():
        qq = q0.detach().requires_grad_(True)
        op(qq).backward(dy)

    ms, peak = measure(fwd_bwd, device, iters)
    min_bytes = 4 * q0.numel() * q0.element_size()  # fwd: read q, write y; bwd: read dy, write dq
    return {"shape": f"{b}×{t}×{h}×{d}", "ms": ms, "peak_mb": peak, "err": err, "min_bytes": min_bytes}


# ---------------------------------------------------------------- driver
BACKENDS = ("naive", "compile", "liger", "torch", "triton", "cuda")
KILN_BACKENDS = {"torch", "triton", "cuda"}  # a failure here fails the run; baselines are only reported


def plan(device: str, dtypes: list[str], kernels: set[str]) -> list[tuple[str, Callable, tuple, set[str]]]:
    """(kernel, fn, shape-args, backends that implement it)."""
    if device == "cuda":
        # the last two log-prob shapes are short batches, where one program per row under-fills the GPU
        lp = [(4096, 32000), (4096, 128256), (2048, 151936), (256, 128256), (64, 151936)]
        rn = [(16384, 4096), (8192, 8192)]
        rp = [(4, 4096, 32, 128)]
    else:
        lp = [(256, 4096), (256, 32000)]
        rn = [(2048, 1024)]
        rp = [(2, 256, 8, 64)]
    out = []
    for dt in dtypes:
        if "logprob" in kernels:
            out += [
                (
                    "logprob+entropy",
                    case_logprob,
                    (n, v, DT[dt]),
                    {"naive", "compile", "liger", "torch", "triton", "cuda"},
                )
                for n, v in lp
            ]
        if "rmsnorm" in kernels:
            out += [
                (
                    "rmsnorm",
                    case_rmsnorm,
                    (n, h, DT[dt]),
                    {"naive", "compile", "liger", "torch", "triton", "cuda"},
                )
                for n, h in rn
            ]
        if "rope" in kernels:
            out += [("rope", case_rope, (*s, DT[dt]), {"naive", "compile", "torch", "triton"}) for s in rp]
    return out


def _fmt_ms(x: float) -> str:
    return f"{x:.2f} ms" if x >= 0.1 else f"{x * 1e3:.0f} µs"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles/pipes default to cp1252
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtypes", nargs="+", default=None, help="default: fp32 on CPU, bf16 fp32 on GPU")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--kernels", nargs="+", default=["logprob", "rmsnorm", "rope"])
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.tag or (torch.cuda.get_device_name(0).replace(" ", "-").lower() if device == "cuda" else "cpu")
    dtypes = a.dtypes or (["bf16", "fp32"] if device == "cuda" else ["fp32"])
    iters = a.iters or (20 if device == "cuda" else 5)
    backends = available_backends(device)
    copy_bw = copy_bandwidth_gbs(device)
    spec_bw = theoretical_bandwidth_gbs() if device == "cuda" else None

    rows = []
    for kernel, fn, args, impls in plan(device, dtypes, set(a.kernels)):
        for be in BACKENDS:
            if be not in impls or be not in a.backends:
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
            os.environ["KILN_KERNEL_BACKEND"] = {"naive": "torch", "compile": "torch", "liger": "torch"}.get(
                be, be
            )
            print(f"{kernel} {args} {be}", file=sys.stderr, flush=True)
            try:
                r = fn(*args, device=device, backend=be, iters=iters)
            except Exception as e:  # report, do not hide
                r = {"error": f"{type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"}
            finally:
                if device == "cuda":
                    torch.cuda.empty_cache()
            r.update({"kernel": kernel, "dtype": str(args[-1]).replace("torch.", ""), "backend": be})
            if "err" in r:
                tol = TOL[args[-1]]
                r["tolerance"] = tol
                r["pass"] = all(
                    v <= tol * (10 if k in ("grad", "dx", "dw", "dq") else 1) for k, v in r["err"].items()
                )
            if "ms" in r and r.get("min_bytes"):
                r["gbs"] = r["min_bytes"] / (r["ms"] / 1e3) / 1e9
                if copy_bw:
                    r["pct_copy_bw"] = 100 * r["gbs"] / copy_bw
                if spec_bw:
                    r["pct_spec_bw"] = 100 * r["gbs"] / spec_bw
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
        "copy_bandwidth_gbs": copy_bw,
        "spec_bandwidth_gbs": spec_bw,
        "triton_autotune": os.environ.get("KILN_TRITON_AUTOTUNE", "1") != "0",
        "command": " ".join(["python", "scripts/bench_kernels.py", *sys.argv[1:]]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"kernels_{tag}.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Kernel comparison — `{tag}`",
        "",
        f"Generated {meta['date']} by `{meta['command']}` on **{meta['device']}** (torch {meta['torch']}, CUDA {meta['cuda']}).",
        f"Forward + backward, median of {iters} iterations. Errors are max |Δ| against a float64 reference;",
        "gradients are held to 10× the output tolerance. *naive* is the unfused PyTorch formula, *compile* is",
        "`torch.compile` of it, *liger* is Liger Kernel; *torch* is Kiln's chunked fallback, *triton* and *cuda* are",
        f"Kiln's fused kernels. Raw data: `kernels_{tag}.json`.",
        "",
    ]
    if copy_bw:
        bw = f"**Bandwidth.** GB/s is the minimum bytes each operation must move divided by its time. This GPU's measured copy bandwidth is **{copy_bw:.0f} GB/s**"
        bw += f" (spec sheet: {spec_bw:.0f} GB/s)" if spec_bw else ""
        bw += "; the % column is relative to the measured copy, the practical ceiling for a memory-bound kernel."
        lines += [bw, ""]
    skipped = {b: why for b, why in backends.items() if why}
    if skipped:
        lines += [
            "Not run on this machine: " + "; ".join(f"**{b}** ({why})" for b, why in skipped.items()) + ".",
            "",
        ]
    if any(r.get("backend") == "liger" and "note" in r for r in rows):
        lines += [
            "*Liger log-prob rows compute cross-entropy only (no entropy) and overwrite the logits with the gradient,",
            "so they do less work and need no gradient buffer; they are a reference point, not a like-for-like match.*",
            "",
        ]
    lines += [
        "| kernel | shape | dtype | backend | time | vs naive | vs compile | GB/s | % of copy BW | peak extra memory | max error (outputs / grad) | ok |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ref = {}
    for r in rows:
        if "ms" in r and r["backend"] in ("naive", "compile"):
            ref[(r["kernel"], r["shape"], r["dtype"], r["backend"])] = r["ms"]
    for r in rows:
        if "skipped" in r:
            continue
        if "error" in r:
            lines.append(
                f"| {r['kernel']} | | {r['dtype']} | {r['backend']} | | | | | | | **{r['error']}** | ❌ |"
            )
            continue
        key = (r["kernel"], r["shape"], r["dtype"])
        vs = []
        for b in ("naive", "compile"):
            base = ref.get((*key, b))
            vs.append("—" if base is None or b == r["backend"] else f"{base / r['ms']:.2f}×")
        errs = " / ".join(f"{v:.1e}" for v in r["err"].values())
        mem = "n/a" if r["peak_mb"] is None else f"{r['peak_mb']:.0f} MB"
        gbs = f"{r['gbs']:.0f}" if "gbs" in r else "—"
        pct = f"{r['pct_copy_bw']:.0f}%" if "pct_copy_bw" in r else "—"
        lines.append(
            f"| {r['kernel']} | {r['shape']} | {r['dtype']} | {r['backend']} | {_fmt_ms(r['ms'])} | {vs[0]} | {vs[1]} | "
            f"{gbs} | {pct} | {mem} | {errs} | {'✅' if r['pass'] else '❌'} |"
        )
    if device == "cpu":
        lines += [
            "",
            "On CPU only the reference paths run, so this table establishes **correctness** of the reference",
            "implementations and the harness itself; timings are CPU timings and say nothing about the GPU kernels.",
            "Run the same command on a CUDA machine to fill in the GPU rows, bandwidth, memory and speed.",
        ]
    md = OUT / f"kernels_{tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    kiln_rows = [r for r in rows if r.get("backend") in KILN_BACKENDS]
    ok = all(r.get("pass", True) for r in kiln_rows if "err" in r) and not any(
        "error" in r for r in kiln_rows
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
