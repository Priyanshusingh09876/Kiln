# Sync vs async GRPO — `cpu-sim`

Generated 2026-10-26 by `python scripts/bench_sync_vs_async.py --steps 60 --seeds 0 1 2 --tag cpu-sim`.
Setup: CPU scheduling simulation: tiny model, 4 mock engines, decode latency 0.15 ms/token, response lengths lognormal(median 600, sigma 1). Machine: Linux x86_64, 2 torch threads, Python 3.13.17, torch 2.14.1+cu130; GPU: none × 0.
60 optimizer steps per run, seeds [0, 1, 2]; mean ± std over seeds. Raw data: `sync_vs_async_cpu-sim.json`.

| mode | wall time | trainer busy | trained tok/s | time to reward ≥ 0.8 | final reward | mean staleness | KL to ref | max IS weight | dropped | GPU util | GPU mem peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sync | 73.0 ± 4.2 s | 4 ± 0% | 210 ± 12 | 30.5 ± 3.1 s | 0.991 ± 0.002 | 0.00 ± 0.00 | 5.4411 ± 0.7983 | 1.00 ± 0.00 | 0 ± 0 | n/a | n/a |
| async-d2 | 34.7 ± 1.2 s | 24 ± 2% | 442 ± 15 | 16.0 ± 3.2 s | 0.979 ± 0.012 | 1.48 ± 0.06 | 4.9766 ± 0.2260 | 1.04 ± 0.01 | 45 ± 1 | n/a | n/a |
| async-d4 | 26.1 ± 2.0 s | 33 ± 1% | 589 ± 45 | 12.7 ± 2.7 s | 0.959 ± 0.022 | 2.82 ± 0.14 | 5.5571 ± 1.7960 | 1.05 ± 0.02 | 15 ± 3 | n/a | n/a |

**Against sync** (same number of steps, same batch size, same seeds):

- **async-d2**: finishes 2.10× faster and reaches reward 0.8 **1.9× sooner** in wall time; final reward 0.979 vs 0.991; 43% of generated batches were dropped as too stale.
- **async-d4**: finishes 2.80× faster and reaches reward 0.8 **2.4× sooner** in wall time; final reward 0.959 vs 0.991; 20% of generated batches were dropped as too stale.

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
