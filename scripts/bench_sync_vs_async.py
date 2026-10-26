#!/usr/bin/env python
"""Synchronous vs asynchronous GRPO on identical hardware, config and seeds.

    # CPU scheduling simulation (no GPU needed, ~3 min): engines are the in-process mock
    # with a heavy-tailed decode-latency model standing in for GPU generation time
    python scripts/bench_sync_vs_async.py --tag cpu-sim

    # real GPUs: any Kiln config (vLLM/SGLang engines); only async.* is varied between runs
    python scripts/bench_sync_vs_async.py --config configs/grpo_math_qwen3_8b.yaml --steps 200 --tag h100x12

For each mode it records: wall time, trainer busy fraction (share of wall time the
trainer is training rather than waiting for rollouts), response tokens trained per
second, mean staleness, wall time to reach a reward threshold, final reward, KL to
the reference, max importance weight, dropped batches, and GPU utilisation / peak
memory when NVML or nvidia-smi is available ("n/a" otherwise, never a guess).

Writes ``docs/benchmarks/sync_vs_async_<tag>.{json,md}``. Every number in the
markdown comes from the JSON written in the same run.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from kiln.config.schema import KilnConfig
from kiln.orchestration import AsyncController
from kiln.utils.gpu_monitor import GPUMonitor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "benchmarks"

MODES = {
    "sync": {"enabled": False, "queue_depth": 1, "max_staleness": 0, "adaptive_depth": False},
    "async-d2": {"enabled": True, "queue_depth": 2, "max_staleness": 2, "adaptive_depth": False},
    "async-d4": {"enabled": True, "queue_depth": 4, "max_staleness": 4, "adaptive_depth": False},
}


def first_byte_reward(items):
    """Dense synthetic reward for the CPU run: first byte of the response close to '7'."""
    r = [max(0.0, 1.0 - abs(ord(it["response"][:1] or "\0") - ord("7")) / 128) for it in items]
    return r, [{"first7": v} for v in r]


def cpu_sim_config(steps: int, engines: int, ms_per_token: float) -> dict:
    return {
        "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2},
        "train": {
            "total_steps": steps,
            "rollout_batch_size": 64,
            "micro_batch_size": 32,
            "optim": {"lr": 1e-2, "warmup_steps": 2},
        },
        "rollout": {
            "engine": "mock",
            "num_engines": engines,
            "sampling": {"n": 8, "max_new_tokens": 4},
            "mock_latency_ms_per_token": ms_per_token,
            "mock_median_tokens": 600,
            "mock_length_sigma": 1.0,
        },
        "algorithm": {"name": "grpo", "kl": {"coef": 0.001}},
        "checkpoint": {"interval": 0},
        "logging": {"backend": "stdout", "log_dir": "/tmp/kiln-bench-sva"},
    }


def time_to_threshold(history: list[dict], thr: float, window: int = 5) -> float | None:
    r = [h.get("reward/mean", 0.0) for h in history]
    for i in range(window - 1, len(r)):
        if sum(r[i - window + 1 : i + 1]) / window >= thr:
            return history[i]["time/wall_s"]
    return None


def run_mode(base: dict, mode: str, seed: int, rows, reward_fn) -> dict:
    raw = json.loads(json.dumps(base))
    raw["async"] = {**raw.get("async", {}), **MODES[mode]}
    raw.setdefault("train", {})["seed"] = seed
    raw.setdefault("logging", {})["run_name"] = f"sva_{mode}_s{seed}"
    cfg = KilnConfig.model_validate(raw)
    torch.manual_seed(seed)
    ctl = AsyncController.from_config(cfg, rows=rows, reward_fn=reward_fn)
    mon = GPUMonitor().start()
    t0 = time.perf_counter()
    res = ctl.run()
    wall = time.perf_counter() - t0
    gpu = mon.stop()
    h = res.history
    tokens = sum(m.get("throughput/tokens_per_s", 0.0) * m.get("time/step_s", 0.0) for m in h)
    last = h[-10:]
    return {
        "mode": mode,
        "seed": seed,
        "steps": res.steps,
        "wall_s": wall,
        "steps_per_s": res.steps / wall,
        "trainer_busy_frac": h[-1]["perf/trainer_busy_frac_cum"],
        "trained_tokens_per_s": tokens / wall,
        "mean_staleness": statistics.fmean(m.get("async/mean_staleness", 0.0) for m in h),
        "final_reward": statistics.fmean(m.get("reward/mean", 0.0) for m in last),
        "kl_ref_final": statistics.fmean(m.get("kl/ref", 0.0) for m in last),
        "is_weight_max": max(m.get("async/is_weight", 1.0) for m in h),
        "dropped_batches": h[-1].get("buffer/dropped_batches", 0.0),
        "reward_curve": [m.get("reward/mean", 0.0) for m in h],
        "wall_curve": [m.get("time/wall_s", 0.0) for m in h],
        **gpu,
    }


def write_report(meta: dict, runs: list[dict], a) -> None:

    lines = [
        f"# Sync vs async GRPO — `{a.tag}`",
        "",
        f"Generated {meta['date']} by `{meta['command']}`.",
        f"Setup: {meta['kind']}. Machine: {meta['machine']}; GPU: {meta['cuda'] or 'none'} × {meta['gpus']}.",
        f"{a.steps} optimizer steps per run, seeds {a.seeds}; mean ± std over seeds. Raw data: `sync_vs_async_{a.tag}.json`.",
        "",
        "| mode | wall time | trainer busy | trained tok/s | time to reward ≥ "
        + f"{a.threshold}"
        + " | final reward | mean staleness | KL to ref | max IS weight | dropped | GPU util | GPU mem peak |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    by_mode = {}
    for mode in a.modes:
        rs = [r for r in runs if r["mode"] == mode]
        agg = {
            k: mean_std([r[k] for r in rs])
            for k in (
                "wall_s",
                "trainer_busy_frac",
                "trained_tokens_per_s",
                "time_to_threshold_s",
                "final_reward",
                "mean_staleness",
                "kl_ref_final",
                "is_weight_max",
                "dropped_batches",
                "gpu_util_mean",
                "gpu_mem_peak_gb",
            )
        }
        by_mode[mode] = agg

        def ms(k, spec=".2f", unit="", scale=1.0, _agg=agg):
            m, s = _agg[k]
            return "n/a" if m is None else f"{m * scale:{spec}} ± {s * scale:{spec}}{unit}"

        reached = sum(1 for r in rs if r["time_to_threshold_s"] is not None)
        ttt = ms("time_to_threshold_s", ".1f", " s") + (
            "" if reached == len(rs) else f" ({reached}/{len(rs)} reached)"
        )
        lines.append(
            f"| {mode} | {ms('wall_s', '.1f', ' s')} | {ms('trainer_busy_frac', '.0f', '%', 100)} | {ms('trained_tokens_per_s', '.0f')} "
            f"| {ttt} | {ms('final_reward', '.3f')} | {ms('mean_staleness', '.2f')} | {ms('kl_ref_final', '.4f')} "
            f"| {ms('is_weight_max', '.2f')} | {ms('dropped_batches', '.0f')} | {ms('gpu_util_mean', '.0f', '%')} | {ms('gpu_mem_peak_gb', '.1f', ' GB')} |"
        )
    if "sync" in by_mode:
        s_wall = by_mode["sync"]["wall_s"][0]
        s_ttt = by_mode["sync"]["time_to_threshold_s"][0]
        lines += ["", "**Against sync** (same number of steps, same batch size, same seeds):", ""]
        for mode in a.modes:
            if mode == "sync":
                continue
            m = by_mode[mode]
            ttt = (
                f"reaches reward {a.threshold} **{s_ttt / m['time_to_threshold_s'][0]:.1f}× sooner** in wall time"
                if s_ttt and m["time_to_threshold_s"][0]
                else "time-to-threshold not comparable (not reached in all runs)"
            )
            dropped = m["dropped_batches"][0] or 0.0
            waste = dropped / (dropped + a.steps)
            lines.append(
                f"- **{mode}**: finishes {s_wall / m['wall_s'][0]:.2f}× faster and {ttt}; final reward "
                f"{m['final_reward'][0]:.3f} vs {by_mode['sync']['final_reward'][0]:.3f}; "
                f"{waste:.0%} of generated batches were dropped as too stale."
            )
        lines += [
            "",
            "Reading the table:",
            "",
            "- *Trainer busy* is the share of wall time the trainer spends training rather than waiting for rollouts.",
            "  In sync mode every step waits for the slowest response across all engines; that wait is what async removes.",
            "- Staleness costs a little learning per step (final reward is slightly lower at a fixed step count), which is",
            "  why the fair comparison is reward against *wall time*, not against steps.",
            "- *Dropped* batches are rollout compute thrown away because they aged past `max_staleness` while their",
            "  worker waited to hand them over. With more engines than queue slots this is large; keep",
            "  `async.queue_depth` ≥ `rollout.num_engines` (compare the two async rows).",
            "- *KL to ref* is similar across modes; it is large here because the toy reward drives the policy",
            "  to a near-deterministic output, not because of asynchrony.",
        ]
    if meta["kind"].startswith("CPU"):
        lines += [
            "",
            "**What this does and does not measure.** Generation time here is a latency model, not a GPU: each",
            "engine call sleeps for its longest simulated response, releasing the GIL so training really runs",
            "concurrently, as it would with engines on separate GPUs. That isolates the *scheduling* effect —",
            "how much trainer time the synchronous loop loses to the length tail and how much async recovers —",
            "together with the *correctness* side (same reward reached, bounded staleness and IS weights).",
            "It says nothing about kernel speed, MFU or vLLM throughput; run with `--config` on GPUs for that.",
        ]
    md = OUT / f"sync_vs_async_{a.tag}.md"
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.2))
        for mode in a.modes:
            r = next(x for x in runs if x["mode"] == mode)
            ax.plot(r["wall_curve"], r["reward_curve"], label=mode)
        ax.set_xlabel("wall-clock seconds")
        ax.set_ylabel("reward/mean")
        ax.set_title(f"Reward vs wall time ({a.tag}, seed {a.seeds[0]})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"sync_vs_async_{a.tag}.png", dpi=130)
    except ImportError:
        pass


def fmt(v, spec=".2f", unit=""):
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="Kiln YAML; omit for the CPU scheduling simulation")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--engines", type=int, default=4, help="CPU sim: number of mock engines")
    ap.add_argument("--ms-per-token", type=float, default=0.15, help="CPU sim: simulated decode ms/token")
    ap.add_argument("--threshold", type=float, default=0.8, help="reward for time-to-threshold")
    ap.add_argument("--tag", default="cpu-sim")
    ap.add_argument("--from-json", type=Path, help="regenerate the report from a previous run's JSON")
    a = ap.parse_args()
    if a.from_json:
        data = json.loads(a.from_json.read_text())
        a.tag, a.steps, a.seeds = data["meta"]["tag"], data["meta"]["steps"], data["meta"]["seeds"]
        a.threshold = data["meta"].get("threshold", a.threshold)
        a.modes = list(dict.fromkeys(r["mode"] for r in data["runs"]))
        write_report(data["meta"], data["runs"], a)
        return 0

    if a.config:
        base = KilnConfig.from_yaml(a.config).model_dump(by_alias=True, mode="json")
        base["train"]["total_steps"] = a.steps
        rows, reward_fn, kind = None, None, f"real engines from `{a.config}`"
    else:
        base = cpu_sim_config(a.steps, a.engines, a.ms_per_token)
        rows = [{"prompt": f"prompt {i}"} for i in range(64)]
        reward_fn, kind = (
            first_byte_reward,
            (
                f"CPU scheduling simulation: tiny model, {a.engines} mock engines, decode latency "
                f"{a.ms_per_token} ms/token, response lengths lognormal(median 600, sigma 1)"
            ),
        )

    runs = []
    for mode in a.modes:
        for seed in a.seeds:
            print(f"== {mode} seed {seed}", file=sys.stderr, flush=True)
            runs.append(run_mode(base, mode, seed, rows, reward_fn))
    for r in runs:
        r["time_to_threshold_s"] = time_to_threshold(
            [{"reward/mean": x, "time/wall_s": w} for x, w in zip(r["reward_curve"], r["wall_curve"])],
            a.threshold,
        )

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "tag": a.tag,
        "kind": kind,
        "machine": f"{platform.system()} {platform.machine()}, {torch.get_num_threads()} torch threads, "
        f"Python {platform.python_version()}, torch {torch.__version__}",
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpus": torch.cuda.device_count(),
        "steps": a.steps,
        "seeds": a.seeds,
        "command": " ".join(["python", "scripts/bench_sync_vs_async.py", *sys.argv[1:]]),
        "threshold": a.threshold,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sync_vs_async_{a.tag}.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=2))
    write_report(meta, runs, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
