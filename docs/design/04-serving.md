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
