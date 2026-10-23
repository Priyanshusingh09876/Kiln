# ADR-01: Asynchronous rollout/training with bounded staleness

**Status:** accepted · **Modules:** `kiln.orchestration`, `kiln.algorithms.losses`

## Problem

In synchronous RL post-training (the verl / OpenRLHF default), one optimizer
step waits for the *slowest* rollout in the batch. With reasoning models,
response lengths are heavy-tailed (p50 ≈ 800 tokens, p99 ≈ 8k), so the
trainer GPUs sit idle for most of each step waiting on a handful of long
generations. Measured on an internal 8+4 GPU layout this put trainer
utilisation under 30%.

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
  higher sample efficiency on paper but destabilised training in practice
  past staleness ≈ 4; the bounded buffer is the compromise.
