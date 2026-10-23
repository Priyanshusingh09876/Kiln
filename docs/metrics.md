# Metrics reference

Exported once per optimizer step to `runs/<run>/metrics.jsonl` and the configured backend.

| metric | meaning | when it moves |
|---|---|---|
| `reward/mean`, `reward/<fn>` | mean sequence reward and per-rule breakdown | the thing you are optimising |
| `loss/policy`, `loss/value`, `loss/entropy`, `loss/total` | loss components | `loss/policy` near 0 with high `clipfrac` ⇒ learning rate too high |
| `policy/clipfrac` | fraction of tokens outside the clip range | > 0.3 sustained ⇒ reduce lr or increase `clip_eps` |
| `policy/approx_kl` | ½·E[(logπ−logπ_old)²] | spikes correlate with instability |
| `policy/entropy` | mean response-token entropy | collapse to ~0 ⇒ mode collapse; add `entropy_coef` or raise temperature |
| `kl/ref`, `kl/coef` | KL to the reference policy and the (possibly adaptive) coefficient | `kl/ref` runaway ⇒ raise `kl.coef` or set `kl.target` |
| `async/mean_staleness` | mean policy-version lag of the consumed batch | should hover near `queue_depth − 1` |
| `async/is_weight` | mean truncated importance weight | far from 1 ⇒ staleness too high for this lr |
| `async/queue_depth` | current (adaptive) buffer capacity | tracks `ceil(train_s / rollout_s)` |
| `buffer/fill` | buffer occupancy | ~1.0 is healthy (trainer is the bottleneck, never starved) |
| `buffer/dropped_batches` | batches discarded for exceeding `max_staleness` | > 0 ⇒ rollout compute wasted; lower `queue_depth` or raise `max_staleness` |
| `buffer/get_wait_s` | trainer time blocked waiting for a batch | > 0 ⇒ rollouts are the bottleneck; add engines |
| `rollout/gen_s`, `rollout/resp_tok_per_s` | per-batch generation time and engine throughput | |
| `rollout/truncated_frac` | responses that hit `max_new_tokens` | > 0.1 ⇒ raise `max_new_tokens` or add `length_penalty` |
| `sync/seconds`, `sync/gbps` | weight sync duration and effective bandwidth | `nccl` transport should show hundreds of Gbps on NVLink |
| `perf/mfu` | model FLOPs utilisation incl. forward-only passes | see `kiln/utils/mfu.py` |
| `throughput/tokens_per_s` | response tokens trained per second | |
| `time/old_logp_s`, `time/ref_logp_s`, `time/fwd_bwd_s`, `time/optim_s`, `time/step_s` | step breakdown | `old_logp + ref_logp` ≈ ⅓ of `fwd_bwd` for a dense model is expected |
| `ckpt/copy_s` | blocking device→host copy time of the last checkpoint | the only checkpoint cost the GPU sees |
