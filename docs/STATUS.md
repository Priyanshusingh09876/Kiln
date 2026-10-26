# Implementation status

What has been run, what has only been written, and what does not exist yet. This
page is the source of truth; if the README and this page disagree, this page wins
and the README is wrong.

**Legend**

- ✅ **Tested** — executed by the test suite on every CI run (CPU), with assertions on behaviour, not just imports.
- 🟡 **Written, not yet executed** — the code exists and is type-checked/linted, but has never run on the hardware it targets. Expect bugs.
- ⬜ **Not implemented** — planned; the config validator rejects it where it would otherwise be silently wrong.

Hardware used for everything marked ✅: a CPU-only Linux container (2–4 cores). **No GPU run of any kind has been done yet.** `docs/VALIDATION.md` is the plan for changing that, and results will be published in `docs/benchmarks/` as they come in.

## Training

| Component | Status | Evidence / notes |
|---|---|---|
| GRPO, Dr. GRPO, RLOO, REINFORCE++ | ✅ | unit tests on advantages/losses; `test_e2e.py` runs the full loop and asserts the reward rises |
| PPO with learned critic + GAE | ✅ | runs end to end (`test_ppo_with_critic_runs`); learning not benchmarked |
| DPO / IPO / hinge (offline) | ✅ | `test_dpo.py`: learns a preference, exact resume, CLI |
| Async loop: bounded-staleness buffer, IS correction, adaptive depth | ✅ | `test_e2e.py`, `test_orchestration.py`; sync-vs-async results in `docs/benchmarks/sync_vs_async_cpu-sim.md` |
| Data-parallel FSDP2 (`parallel.dp`) | ✅ on CPU/gloo · 🟡 on CUDA/NCCL | `test_multirank.py`: 2 real processes, sharded DTensor params, gradient sync verified, rank-sharded checkpoint, bit-exact resume. Same code path on GPU, but bf16 mixed precision + NCCL have not been run |
| Tensor / pipeline / context parallelism in the trainer | ⬜ | the device mesh creates the groups, but no model code uses them; configs with `tp/pp/cp > 1` are rejected |
| MoE expert parallelism | ⬜ | |
| Sequence packing (FFD + block-diagonal mask) | ✅ | packed forward equals padded forward (`test_packed_forward_matches_padded_forward`) |
| Activation checkpointing | 🟡 | applied to HF models only; the tiny test model does not use it |
| HuggingFace model loading (`model.name_or_path: <hf id>`) | 🟡 | `transformers` is not installed in CI; only the built-in tiny model has been trained |
| Async atomic checkpointing | ✅ | atomic publish, mutation-during-write, writer-error surfacing, pruning; multi-rank via a dedicated gloo group |
| Exact resume (weights, optimizer, scheduler, data position, sampler RNG) | ✅ sync · ✅ async (approximate) | sync: resumed run bit-identical to uninterrupted (1 and 2 ranks). Async: batches in flight at checkpoint time are skipped; `test_async_resume_completes_and_respects_staleness` |

## Rollouts and weight sync

| Component | Status | Evidence / notes |
|---|---|---|
| Mock engine (tiny model, in-process) | ✅ | used by all CPU tests; optional decode-latency model for scheduling simulations |
| vLLM engine + in-worker extension (`KilnWorkerExtension`) | 🟡 | never run; written against vLLM's `worker_extension_cls` / `collective_rpc` API, which changes between releases |
| SGLang HTTP engine | 🟡 | never run |
| Weight sync: in-process transport | ✅ | `test_orchestration.py`, resume tests |
| Weight sync: CPU/Ray transport | 🟡 | |
| Weight sync: NCCL broadcast into live engine workers | 🟡 | never run; the design is in ADR 03 |
| Partial rollouts (pause → swap → resume) | 🟡 | the mock engine's pause is a no-op, so only the API is exercised |
| Ray runtime (placement groups, colocated/disaggregated) | 🟡 | never run; Ray is not installed in CI |

## Kernels

| Component | Status | Evidence / notes |
|---|---|---|
| PyTorch reference paths (log-prob+entropy, RMSNorm, RoPE, FP8 dequant GEMM) | ✅ | fwd + bwd vs. naive and vs. float64 (`test_kernels.py`, `docs/benchmarks/kernels_cpu.md`) |
| Triton kernels | 🟡 | parity tests exist and skip without CUDA; never executed |
| CUDA C++ kernels (`csrc/*.cu`) | 🟡 type-checked | compiled by g++ against real libtorch + CUDA headers in CI (catches compile errors); never compiled by nvcc or executed |
| Backend pinning (`KILN_KERNEL_BACKEND`) | ✅ | `test_backend_forcing_actually_disables_triton` |
| Kernel benchmark harness | ✅ on CPU | `scripts/bench_kernels.py`; Triton/CUDA rows need a GPU run |

## Rewards and serving

| Component | Status | Evidence / notes |
|---|---|---|
| Rule verifiers (math, format, repetition, length, Python unit tests) | ✅ | `test_data_and_rewards.py` |
| Reward HTTP server + client | ✅ | FastAPI test client |
| Reward-model scorer | 🟡 | needs `transformers` and a model |
| Prefix-cache router (Python + C++ core) | ✅ | C++ differential-tested against Python; under load in `docs/benchmarks/gateway_demo.md` |
| Scheduler / capacity simulator | ✅ | unit tests; it is a model, calibrate before trusting absolute numbers |
| Canary controller (matched control, p50/p99 ratios, warm-up) | ✅ | unit tests + three load scenarios in `docs/benchmarks/gateway_demo.md` |
| Gateway (real HTTP, simulated replicas) | ✅ | `test_gateway_load.py`; against real vLLM replicas: 🟡 |

## Operations

| Component | Status | Evidence / notes |
|---|---|---|
| CPU CI (lint, tests on 2 Pythons, 2-rank FSDP, CUDA host type-check, perf gate) | ✅ | `.github/workflows/ci.yml` |
| GPU CI job | 🟡 defined, not running | needs a self-hosted GPU runner registered to the repo; none is attached |
| Docker images, KubeRay job, Slurm scripts | 🟡 | never deployed |
| Multi-node runs | ⬜ not validated | plan in `docs/VALIDATION.md` |

## Bugs found while closing these gaps

Recorded because they are what happens to infrastructure that has only run on one machine with one process:

1. `shard_model` skipped FSDP whenever CUDA was absent, so multi-process CPU runs trained each rank independently without syncing gradients — silently.
2. The async checkpointer's writer thread issued barriers on the training process group; collectives from two threads on one group can interleave and deadlock. Now uses a dedicated gloo group.
3. Rank-sharded checkpoints could not be loaded back into FSDP2 (DTensor) parameters or optimizer state.
4. Every rank seeded its rollout workers identically, so data-parallel ranks trained on duplicate prompts.
5. After a resume, rollout engines still held the *initial* weights tagged version 0: in async mode every new batch was dropped as stale and the run hung forever.
6. Resume restarted the prompt stream from the beginning.
7. Mock engines deep-copied the *sharded* trainer model; generation would have issued FSDP collectives at different times on different ranks and deadlocked.
8. The reference policy was copied from the sharded state dict.
9. Synchronous mode generated each batch on a single engine instead of splitting it across engines, which would have overstated the async speed-up.
10. The gateway only re-evaluated the canary when `/healthz` was called, so a bad canary under load was never rolled back — and every health probe could *promote* it.
11. The canary SLO compared only p99s, so a regression that shifted the whole latency distribution hid in a noisy baseline tail; it now also compares medians.
12. Judging a canary against the warm fleet rolled back a *healthy* canary because of its cold prefix cache; fixed with a matched control arm.
13. `KILN_KERNEL_BACKEND=torch` did not disable Triton, so a "torch vs Triton" benchmark would have compared Triton with itself.
14. Configs accepted `tp/pp/cp > 1` although nothing implements them; the shipped 70B config relied on it.
15. DPO had a loss function and a dataset but no training loop.

Fixes 1, 3–7 and 9–15 have tests that target them directly; for 5, 6 and 10 I confirmed the test fails when
the fix is reverted. Fixes 2 and 8 are exercised by the two-rank test but no test isolates them.
