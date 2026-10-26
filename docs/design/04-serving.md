# ADR-04: Serving control plane — routing, scheduling, canaries

**Status:** accepted · **Module:** `kiln.serving`

The data plane (vLLM / SGLang / TensorRT-LLM) is someone else's problem; this
module is the control plane that sits around N replicas of it. Everything
here is pure Python with no model dependency so it can be unit-tested and
used as a simulator.

## Prefix-cache-aware router (`router.py`)

Each replica keeps a block-granular radix tree mirroring the prefixes it has
been sent (LRU-bounded to the replica's KV capacity). A request is scored
per replica as

```
score = matched_prefix_tokens · cache_weight − in_flight · load_weight
```

with a hard imbalance cap so a hot system prompt cannot pin all traffic to
one replica. Multi-turn chats and shared system prompts land on the replica
that already holds their KV blocks; `kiln_replica_hit_rate` exposes the
result per replica.

## Scheduler (`scheduler.py`)

A faithful model of a continuous-batching engine's admission logic:

* paged KV with a block allocator and a decode reservation;
* **chunked prefill** so a 32k-token prompt is interleaved with decode
  steps instead of stalling every in-flight stream for one tick;
* **preemption** of the most recently admitted request on KV exhaustion
  (recompute on resume) — evicting the youngest wastes the least work.

`simulate()` drives it with Poisson arrivals and a simple cost model and
returns TTFT/E2E percentiles, throughput, preemption count and peak KV
utilisation. `examples/03_serving_capacity_planning.py` bisects on replica
count to answer "how many replicas for X RPS under a p99 SLO". The cost
model constants are placeholders — calibrate `prefill_tok_us` /
`decode_step_us` from one real profile of your engine before trusting the
absolute numbers; the *relative* conclusions (knee of the curve, effect of
chunking, KV sizing) hold regardless.

## Canary guardrails (`canary.py`)

A new checkpoint is promoted through a traffic ramp (1% → 5% → 25% → 50% →
100%). At each level the controller compares the canary's sliding window
against the baseline on absolute p99, p99 relative to baseline, error rate,
and an optional quality score (e.g. reward-model agreement), and only
promotes after `min_samples`. Any breach trips the canary to 0% and stays
tripped until a human resets it — automatic re-promotion after a transient
is how you ship a bad model twice.

The gateway (`gateway.py`) wires router + canary behind an
OpenAI-compatible `/v1/completions` and exports Prometheus metrics.

## What the load test changed

Running the gateway under real HTTP load (`examples/04_gateway_canary_demo.py`,
results in `docs/benchmarks/gateway_demo.md`) found four problems in the first
version of this design:

1. **Evaluation was only triggered by `/healthz`.** Under load with no health
   checks a bad canary was never rolled back, and every health probe could
   *promote* it. Evaluation is now driven by canary traffic (every
   `eval_every` canary requests) and health endpoints are read-only.
2. **p99-only comparison hid real regressions.** A canary whose whole latency
   distribution shifted could stay under 1.5× a noisy baseline p99. The SLO
   also compares medians (`max_p50_ratio`).
3. **Judging against the warm fleet rolled back healthy builds.** A new
   replica starts with a cold prefix cache and, at a small traffic share, sees
   a random sample of all tenants, while fleet replicas each own a
   cache-friendly slice. The fix is a *matched control* (the Kayenta pattern): a
   fresh replica of the current build that starts with the canary and gets the
   same traffic share; the canary is judged only against it.
4. **Prefix routing trades balance for hit rate.** It raised the cache hit rate
   from 53% to 89% but let the busiest replica take 37% of traffic instead of
   25%. `max_imbalance` is the knob, and it should be set from measured
   capacity, not left at its default.
