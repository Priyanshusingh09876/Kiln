# ADR-01: Asynchronous rollout/training with bounded staleness

**Status:** accepted · **Modules:** `kiln.orchestration`, `kiln.algorithms.losses`

## Problem

In synchronous RL post-training (the verl / OpenRLHF default), one optimizer
step waits for the *slowest* rollout in the batch. With reasoning models,
response lengths are heavy-tailed (a p99 ten times the median is typical), so the
trainer GPUs sit idle for most of each step waiting on a handful of long
generations. In the CPU scheduling simulation below, the synchronous
trainer was busy only 4% of wall time; on real GPUs the fraction depends on
model size and length distribution and has not been measured yet.

## Decision

Decouple the two with a **bounded, version-aware experience buffer**:

```
rollout workers ──put(batch, v)──▶ ExperienceBuffer ──get(current_v)──▶ trainer
                                   capacity = queue_depth
                                   drops batches with (current_v − v) > max_staleness
```

* Rollout engines run continuously and tag every sample with the policy
  version that produced it.
* The trainer consumes the oldest sufficiently-fresh batch; anything older
  than `max_staleness` versions is dropped and counted.
* After each optimizer step the new weights are pushed to the engines
  (`WeightSyncer`). With `partial_rollout=true` the engines are paused at a
  token boundary, swapped, and resumed, so no in-flight generation is wasted.

### Correctness: importance weighting

Samples are now off-policy by up to `max_staleness` versions. The clipped
surrogate uses the ratio `π_θ / π_old` where `π_old` is recomputed on the
*trainer* at step start — that part is standard PPO. The remaining mismatch
`π_old / π_behaviour` (trainer weights vs. the engine weights that actually
sampled) is corrected with truncated importance weights:

```
w_t = min( exp(logπ_old(a_t) − logπ_beh(a_t)), is_clip )      # token mode
w   = min( exp( mean_t [logπ_old − logπ_beh] ), is_clip )      # sequence mode
loss = aggregate( w · clipped_surrogate )
```

Token mode is the default. Sequence mode (GSPO-style) is used for MoE where
per-token ratios are dominated by router noise. The engine's reported
log-probs are the behaviour policy regardless of partial rollouts, so the
weights remain correct when a sample spans two versions.

`tests/test_e2e.py::test_sync_mode_is_deterministic_and_on_policy` checks
that with `max_staleness=0` every weight is exactly 1, and
`test_importance_weights_depart_from_one_when_stale` checks they are not
when staleness is allowed.

### Adaptive depth

The right queue depth is `ceil(T_train / T_rollout)`, which changes as
responses get longer during training. `StalenessController` keeps EMAs of
both and resizes the buffer (never above `max_staleness`), one step at a
time with a cooldown.

## Consequences

* Trainer never waits on the tail of the length distribution; the price is a
  controlled amount of off-policy-ness, which the IS weights and clipping
  absorb.
* `buffer/dropped_batches` is the key operational metric: non-zero means
  rollout capacity is wasted and depth or staleness should be tuned.
* Sync mode (`async.enabled=false`) remains available for exact-reproducibility
  runs and is what the determinism tests use.

## Alternatives considered

* **Drain-and-sync (verl default):** simplest, but leaves the trainer idle
  on the length tail. Kept as the `partial_rollout=false` path.
* **Fully off-policy replay buffer (large, reused across many versions):**
  higher sample efficiency on paper, but importance weights become
  high-variance as staleness grows and truncating them biases the gradient;
  the bounded buffer keeps both small. Where the useful limit lies for real
  models is an open question for the GPU runs.

## Measured (CPU scheduling simulation)

`docs/benchmarks/sync_vs_async_cpu-sim.md`: with 4 engines and lognormal response
lengths, async reached the target reward 1.9× (queue depth 2) and 2.4× (depth 4)
sooner in wall time than a synchronous loop that splits each batch across all four
engines, at slightly lower reward per step. Two findings from that run shaped the
guidance above:

* **Queue depth must be at least the number of engines.** With depth 2 and 4
  engines, 43% of generated batches were dropped as stale: finished batches aged
  while their workers waited for a free slot. Depth 4 cut that to 20%.
* **Compare reward against wall time, not steps.** Per step, staleness costs a
  little; per second, async wins. A benchmark that plots reward per step will
  always make async look worse.

## Resume semantics

A checkpoint stores, besides weights and optimizer state, each rollout worker's
position in its (seeded, deterministic) prompt stream and the engine's sampler
RNG. On resume the engines are re-synced to the restored weights *before* any
rollout. Without that, every new sample is tagged with the initial policy
version, the buffer drops it as stale, and an async run hangs — which is what
happened before `tests/test_resume.py` existed. Synchronous runs resume
bit-identically; asynchronous runs skip the batches that were in flight.
