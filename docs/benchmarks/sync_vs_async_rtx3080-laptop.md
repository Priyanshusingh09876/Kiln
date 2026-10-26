# Sync vs async GRPO — `rtx3080-laptop`

Generated 2026-10-26 by `python scripts/bench_sync_vs_async.py --tag rtx3080-laptop --seeds 0 1 2`.
Setup: CPU scheduling simulation: tiny model, 4 mock engines, decode latency 0.15 ms/token, response lengths lognormal(median 600, sigma 1). Machine: Windows AMD64, 14 torch threads, Python 3.12.3, torch 2.14.1+cu126; GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU × 1.
60 optimizer steps per run, seeds [0, 1, 2]; mean ± std over seeds. Raw data: `sync_vs_async_rtx3080-laptop.json`.

| mode | wall time | trainer busy | trained tok/s | time to reward ≥ 0.8 | final reward | mean staleness | KL to ref | max IS weight | dropped | GPU util | GPU mem peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sync | 74.4 ± 4.2 s | 5 ± 1% | 206 ± 12 | 31.4 ± 2.6 s | 0.991 ± 0.002 | 0.00 ± 0.00 | 5.4411 ± 0.7983 | 1.00 ± 0.00 | 0 ± 0 | 5 ± 2% | 0.3 ± 0.0 GB |
| async-d2 | 39.4 ± 4.4 s | 35 ± 1% | 392 ± 42 | 17.9 ± 3.1 s | 0.981 ± 0.010 | 1.49 ± 0.06 | 5.4671 ± 0.1940 | 1.05 ± 0.03 | 53 ± 2 | 7 ± 0% | 0.3 ± 0.0 GB |
| async-d4 | 32.1 ± 1.4 s | 54 ± 7% | 478 ± 22 | 16.0 ± 1.8 s | 0.958 ± 0.009 | 2.84 ± 0.14 | 7.4229 ± 2.1737 | 1.04 ± 0.01 | 19 ± 2 | 8 ± 1% | 0.3 ± 0.0 GB |

**Against sync** (same number of steps, same batch size, same seeds):

- **async-d2**: finishes 1.89× faster and reaches reward 0.8 **1.8× sooner** in wall time; final reward 0.981 vs 0.991; 47% of generated batches were dropped as too stale.
- **async-d4**: finishes 2.32× faster and reaches reward 0.8 **2.0× sooner** in wall time; final reward 0.958 vs 0.991; 24% of generated batches were dropped as too stale.

Reading the table:

- *Trainer busy* is the share of wall time the trainer spends training rather than waiting for rollouts.
  In sync mode every step waits for the slowest response across all engines; that wait is what async removes.
- Staleness costs a little learning per step (final reward is slightly lower at a fixed step count), which is
  why the fair comparison is reward against *wall time*, not against steps.
- *Dropped* batches are rollout compute thrown away because they aged past `max_staleness` while their
  worker waited to hand them over. With more engines than queue slots this is large; keep
  `async.queue_depth` ≥ `rollout.num_engines` (compare the two async rows).
- *KL to ref* is similar across modes; it is large here because the toy reward drives the policy
  to a near-deterministic output, not because of asynchrony.

**What this does and does not measure.** Generation time here is a latency model, not a GPU: each
engine call sleeps for its longest simulated response, releasing the GIL so training really runs
concurrently, as it would with engines on separate GPUs. That isolates the *scheduling* effect —
how much trainer time the synchronous loop loses to the length tail and how much async recovers —
together with the *correctness* side (same reward reached, bounded staleness and IS weights).
It says nothing about kernel speed, MFU or vLLM throughput; run with `--config` on GPUs for that.
