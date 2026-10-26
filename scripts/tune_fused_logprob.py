#!/usr/bin/env python
"""Tuning sweep for the fused log-prob + entropy kernels: before (v1) vs after.

    python scripts/tune_fused_logprob.py --tag rtx3080-laptop

Times the forward and the backward kernels separately for every configuration in
the search space, on shapes from long training batches to short ones, and reports
achieved bandwidth against this GPU's measured copy bandwidth. Three things are
compared:

* **v1** — the kernels as first benchmarked: one program per row, a fixed block
  size, and a backward that writes an fp32 gradient which is then converted to the
  logits dtype in a separate pass;
* every individual configuration (block size, warps, split factor, row vs tiled
  backward), so the best one is visible rather than asserted;
* **shipped** — what training now runs: ``fused_logprob_entropy`` with
  autotuning, split rows for short batches, and the tiled backward that stores the
  gradient in the logits dtype.

Writes ``docs/benchmarks/tune_logprob_<tag>.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent.parent / "docs" / "benchmarks"
DEVICE = "cuda"  # overridable for a dry run of the sweep logic
DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def bench(fn) -> float:
    from triton.testing import do_bench

    return float(do_bench(fn, warmup=25, rep=100, return_mode="median"))


def copy_bw() -> float:
    n = 512 * 2**20
    a = torch.empty(n, dtype=torch.uint8, device=DEVICE)
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    torch.cuda.empty_cache()
    return 2 * n / (ms / 1e3) / 1e9


def sweep_shape(N: int, V: int, dtype: torch.dtype, bw: float) -> dict:
    import triton

    from kiln.kernels import fused_logprob as F
    from kiln.kernels.dispatch import next_pow2

    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(N, V, generator=g) * 3).to(DEVICE, dtype)
    t = torch.randint(0, V, (N,), generator=g).to(DEVICE)
    glp = torch.randn(N, generator=g).to(DEVICE)
    gh = torch.randn(N, generator=g).to(DEVICE)
    es = x.element_size()
    fwd_bytes = N * V * es
    lp, ent, lse = (torch.empty(N, device=DEVICE) for _ in range(3))

    def gbs(nbytes, ms):
        return nbytes / (ms / 1e3) / 1e9

    def row(kind, config, ms, nbytes):
        return {
            "kind": kind,
            "config": config,
            "ms": ms,
            "gbs": gbs(nbytes, ms),
            "pct": 100 * gbs(nbytes, ms) / bw,
        }

    fwd, bwd = [], []
    # ---- forward: one program per row
    for B in (1024, 2048, 4096, 8192):
        for w in (4, 8, 16):
            ms = bench(
                lambda B=B, w=w: F._fwd_kernel[(N,)](
                    x, t, lp, ent, lse, V, x.stride(0), BLOCK_V=B, num_warps=w
                )
            )
            fwd.append(row("row", f"BLOCK_V={B} warps={w}", ms, fwd_bytes))
    # ---- forward: split rows across programs
    for S in (2, 4, 8, 16, 32, 64):
        for B in (1024, 2048):
            chunk = triton.cdiv(triton.cdiv(V, S), B) * B
            s_eff = triton.cdiv(V, chunk)
            if s_eff != S:
                continue
            parts = [torch.empty(N, S, device=DEVICE) for _ in range(4)]
            for w in (4, 8):

                def run(S=S, B=B, w=w, chunk=chunk, parts=parts):
                    F._fwd_partial_kernel[(N, S)](
                        x, t, *parts, V, x.stride(0), chunk, S, BLOCK_V=B, num_warps=w
                    )
                    F._combine_partials(*parts)

                fwd.append(row("split", f"S={S} BLOCK_V={B} warps={w}", bench(run), fwd_bytes))

    # ---- backward
    Bv1 = min(next_pow2(V), 8192)
    g32 = torch.empty(N, V, device=DEVICE, dtype=torch.float32)

    def v1_bwd():
        F._bwd_kernel[(N,)](
            x, t, lse, ent, glp, gh, g32, V, x.stride(0), g32.stride(0), BLOCK_V=Bv1, num_warps=16
        )
        return g32.to(dtype)

    v1_bwd_bytes = N * V * (es + 4 + 4 + es)  # read x, write fp32, read fp32, write dtype
    bwd.append(row("v1 row + fp32 grad + convert", f"BLOCK_V={Bv1} warps=16", bench(v1_bwd), v1_bwd_bytes))
    gd = torch.empty(N, V, device=DEVICE, dtype=dtype)
    bwd_bytes = 2 * N * V * es
    for B in (1024, 2048, 4096, 8192):
        for w in (4, 8, 16):
            ms = bench(
                lambda B=B, w=w: F._bwd_kernel[(N,)](
                    x, t, lse, ent, glp, gh, gd, V, x.stride(0), gd.stride(0), BLOCK_V=B, num_warps=w
                )
            )
            bwd.append(row("row, direct dtype", f"BLOCK_V={B} warps={w}", ms, bwd_bytes))
            ms = bench(
                lambda B=B, w=w: F._bwd_tile_kernel[(N, triton.cdiv(V, B))](
                    x, t, lse, ent, glp, gh, gd, V, x.stride(0), gd.stride(0), BLOCK_V=B, num_warps=w
                )
            )
            bwd.append(row("tile, direct dtype", f"BLOCK_V={B} warps={w}", ms, bwd_bytes))

    v1_fwd = next(r for r in fwd if r["kind"] == "row" and r["config"] == f"BLOCK_V={Bv1} warps=16")
    v1_bwd_r = bwd[0]
    best_fwd = min(fwd, key=lambda r: r["ms"])
    best_bwd = min(bwd[1:], key=lambda r: r["ms"])

    # ---- end to end through autograd: v1 (reconstructed) vs shipped
    def e2e_shipped():
        xx = x.detach().requires_grad_(True)
        a, b = F.fused_logprob_entropy(xx, t)
        torch.autograd.backward([a, b], [glp, gh])

    os.environ["KILN_KERNEL_BACKEND"] = "triton"
    e2e_shipped()  # autotune happens here, outside the timing
    shipped_ms = bench(e2e_shipped)
    os.environ["KILN_KERNEL_BACKEND"] = "auto"
    split = F.split_factor(N, V, x.device)
    torch.cuda.empty_cache()

    return {
        "N": N,
        "V": V,
        "dtype": str(dtype).replace("torch.", ""),
        "fwd": fwd,
        "bwd": bwd,
        "v1": {"fwd_ms": v1_fwd["ms"], "bwd_ms": v1_bwd_r["ms"], "total_ms": v1_fwd["ms"] + v1_bwd_r["ms"]},
        "best": {"fwd": best_fwd, "bwd": best_bwd, "total_ms": best_fwd["ms"] + best_bwd["ms"]},
        "shipped": {
            "ms": shipped_ms,
            "gbs": gbs(3 * N * V * es, shipped_ms),
            "pct": 100 * gbs(3 * N * V * es, shipped_ms) / bw,
            "split": split,
        },
    }


def write_report(meta: dict, results: list[dict], tag: str, bw: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"tune_logprob_{tag}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8"
    )

    L = [
        f"# Fused log-prob tuning sweep — `{tag}`",
        "",
        f"Generated {meta['date']} by `{meta['command']}` on **{meta['device']}** ({meta['sms']} SMs, torch {meta['torch']}).",
        f"Kernel times are medians from `triton.testing.do_bench`. Measured copy bandwidth: **{bw:.0f} GB/s**; % is relative to it.",
        "GB/s counts the minimum bytes each kernel must move (forward: read logits; backward: read logits, write gradient).",
        "",
        "## Summary: v1 vs best configuration vs what training runs",
        "",
        "| shape | v1 fwd + bwd | best fwd + best bwd | shipped (autograd, autotuned) | shipped % of copy BW | speedup over v1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        L.append(
            f"| {r['N']}×{r['V']} {r['dtype']} | {r['v1']['total_ms']:.2f} ms | {r['best']['total_ms']:.2f} ms | "
            f"{r['shipped']['ms']:.2f} ms (split={r['shipped']['split']}) | {r['shipped']['pct']:.0f}% | "
            f"{r['v1']['total_ms'] / r['shipped']['ms']:.2f}× |"
        )
    L += [
        "",
        "*v1* is the forward with one program per row (`BLOCK_V=min(next_pow2(V), 8192)`, 16 warps) plus a backward",
        "that writes fp32 and converts. *shipped* includes autograd overhead and the split-row merge.",
        "",
    ]
    for r in results:
        L += [
            f"## {r['N']}×{r['V']} {r['dtype']}",
            "",
            "**Forward** (fastest 8, and v1)",
            "",
            "| kind | config | time | GB/s | % copy BW |",
            "|---|---|---|---|---|",
        ]
        top = sorted(r["fwd"], key=lambda x: x["ms"])[:8]
        v1c = f"BLOCK_V={min(1 << (r['V'] - 1).bit_length(), 8192)} warps=16"
        v1row = next(x for x in r["fwd"] if x["kind"] == "row" and x["config"] == v1c)
        for x in top + ([v1row] if v1row not in top else []):
            tagv1 = " (v1)" if x is v1row else ""
            L.append(
                f"| {x['kind']}{tagv1} | {x['config']} | {x['ms']:.3f} ms | {x['gbs']:.0f} | {x['pct']:.0f}% |"
            )
        L += [
            "",
            "**Backward** (fastest 8, and v1)",
            "",
            "| kind | config | time | GB/s | % copy BW |",
            "|---|---|---|---|---|",
        ]
        topb = sorted(r["bwd"][1:], key=lambda x: x["ms"])[:8]
        for x in topb + [r["bwd"][0]]:
            L.append(f"| {x['kind']} | {x['config']} | {x['ms']:.3f} ms | {x['gbs']:.0f} | {x['pct']:.0f}% |")
        L.append("")
    md = OUT / f"tune_logprob_{tag}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dtype", default="bf16", choices=list(DT))
    ap.add_argument(
        "--shapes", nargs="+", default=["4096x128256", "2048x151936", "256x128256", "64x151936"], help="NxV"
    )
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 2
    tag = a.tag or torch.cuda.get_device_name(0).replace(" ", "-").lower()
    bw = copy_bw()
    results = []
    for s in a.shapes:
        N, V = (int(v) for v in s.lower().split("x"))
        print(f"sweeping {N}x{V} {a.dtype}", file=sys.stderr, flush=True)
        results.append(sweep_shape(N, V, DT[a.dtype], bw))

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "device": torch.cuda.get_device_name(0),
        "sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "copy_bandwidth_gbs": bw,
        "command": " ".join(["python", "scripts/tune_fused_logprob.py", *sys.argv[1:]]),
    }
    write_report(meta, results, tag, bw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
