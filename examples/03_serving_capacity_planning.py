"""Use the scheduler simulator to size a deployment before buying GPUs.

Question answered: how many replicas with a given KV budget are needed to hold
TTFT p99 under a target at a given request rate and prompt/response mix?
"""

from __future__ import annotations

from kiln.serving.scheduler import SchedulerConfig, simulate


def _ok(n: int, target_rps: float, ttft_p99_s: float, cfg: SchedulerConfig) -> bool:
    r = simulate(
        cfg, num_requests=400, arrival_rps=target_rps / n, prompt_len=(256, 2048), max_new=(64, 1024), seed=0
    )
    return r.ttft_p99 <= ttft_p99_s and r.throughput_rps * n >= 0.95 * target_rps


def replicas_needed(
    target_rps: float, ttft_p99_s: float, cfg: SchedulerConfig, max_replicas: int = 256
) -> int:
    """Smallest replica count meeting the SLO (SLO satisfaction is monotone in n, so bisect)."""
    lo, hi = 1, 1
    while hi < max_replicas and not _ok(hi, target_rps, ttft_p99_s, cfg):
        lo, hi = hi, hi * 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _ok(mid, target_rps, ttft_p99_s, cfg):
            hi = mid
        else:
            lo = mid + 1
    return hi


if __name__ == "__main__":
    cfg = SchedulerConfig(num_blocks=16384, block_size=16, max_num_seqs=256, max_num_batched_tokens=8192)
    for rps in (50, 200, 1000):
        print(f"{rps:>5} rps, TTFT p99 <= 500ms -> {replicas_needed(rps, 0.5, cfg)} replicas")
