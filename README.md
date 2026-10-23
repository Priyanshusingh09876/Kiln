<p align="center"><img src="docs/kiln.svg" width="620" alt="kiln: async rollout -> experience buffer -> FSDP trainer -> weight sync"></p>

# Kiln

**An asynchronous LLM post-training and serving stack: GRPO / PPO / DPO on FSDP2, with vLLM / SGLang rollouts, fused Triton kernels, and the serving control plane to ship the result.**

[![ci](https://github.com/priyanshu-singh/kiln/actions/workflows/ci.yml/badge.svg)](https://github.com/priyanshu-singh/kiln/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

```bash
pip install -e ".[dev]"
pytest -q                                   # 113 tests, ~40 s on a laptop CPU, no GPU needed
kiln train configs/tiny_cpu_smoke.yaml      # the full async GRPO loop end-to-end on CPU
kiln train configs/grpo_math_qwen3_8b.yaml --ray   # 8 trainer GPUs + 4 vLLM engines
```

## Why this exists

RL post-training has a structural inefficiency: one optimizer step waits for
the *slowest* rollout in the batch, and with reasoning models response
lengths are heavy-tailed. Kiln decouples generation from training with a
bounded, version-aware experience buffer and corrects the resulting
off-policy-ness with truncated importance weights — so trainer GPUs are never
idle on the length tail, and the math stays right. Everything around that
loop (weight sync into live engines, async atomic checkpoints, fused
kernels for the per-token log-prob hot path, a reward server, a
cache-aware inference gateway with SLO-gated canaries) is here because a
training loop without them does not run for three days on 64 GPUs.

## Architecture

```
            prompts (JSONL/Parquet, sharded, seeded)
                         │
      ┌──────────────────┼───────────────────┐
      ▼                  ▼                   ▼
 RolloutWorker 0    RolloutWorker 1     RolloutWorker k     each drives one engine:
 [vLLM tp=1]        [vLLM tp=1]         [SGLang http]       vLLM · SGLang · in-process mock
      │ generate() → samples tagged with policy_version
      │ reward_fn()  → rules / HTTP reward server / reward model
      ▼                  ▼                   ▼
 ┌──────────────────────────────────────────────────┐
 │ ExperienceBuffer   capacity = queue_depth (adaptive)│  put() blocks when full (back-pressure)
 │                    drops batches older than         │  get() returns oldest fresh batch
 │                    max_staleness versions           │  every drop is counted and exported
 └──────────────────────────┬───────────────────────┘
                            ▼
 ┌──────────────────────────────────────────────────┐
 │ PolicyTrainer (FSDP2 dp × TP × PP × CP mesh)     │  1. old_logp / ref_logp (no-grad, packed)
 │   algorithm: GRPO · Dr.GRPO · RLOO · REINFORCE++ │  2. advantages (group stats / GAE)
 │              · PPO (+critic) · DPO (offline)      │  3. clipped surrogate × IS weights, KL, entropy
 │   kernels:   fused logprob+entropy, RMSNorm, RoPE,│  4. clip grads, optimizer step
 │              FP8 block GEMM  (Triton, torch ref)  │  5. policy_version += 1
 └──────────────────────────┬───────────────────────┘
                            ▼
      WeightSyncer  (inproc | CPU/Ray | NCCL broadcast into live vLLM workers, bucketed, LoRA-only option)
                            │  partial rollouts: pause → swap → resume, no drained queue
                            ▼
                     engines keep generating with the new weights
      ╰── every N steps: AsyncCheckpointer (pinned-host copy → background write → atomic rename)
```

## What's inside

| Area | Module | What it does |
|---|---|---|
| Algorithms | `kiln/algorithms` | GRPO, Dr. GRPO, RLOO, REINFORCE++, PPO with GAE + clipped value loss, DPO/IPO/hinge. DAPO clip-higher, dual-clip, token / sequence / length-unbiased loss aggregation, k1/k2/k3 KL, adaptive KL controller, KL-in-reward or KL-in-loss. |
| Async loop | `kiln/orchestration` | Version-aware `ExperienceBuffer`, `RolloutWorker`, `StalenessController` (adaptive queue depth), `AsyncController`; Ray runtime with placement groups, colocated or disaggregated layouts. |
| Trainer | `kiln/trainer`, `kiln/distributed` | FSDP2 `fully_shard` per block with mixed precision and activation checkpointing; `(pp, dp, cp, tp)` device mesh; sequence packing with first-fit-decreasing and block-diagonal masks; MFU accounting that counts the forward-only passes honestly. |
| Rollouts | `kiln/rollout` | Engine contract (`generate / update_weights / pause / resume / sleep / wake`); vLLM engine with an in-worker extension for NCCL weight receive and in-place `load_weights` (KV cache and CUDA graphs survive); SGLang HTTP engine; mock engine for CPU tests. |
| Kernels | `kiln/kernels` | Single-pass online-softmax log-prob + entropy (fwd + bwd, never materialises `[N,V]` probs), RMSNorm, RoPE with arbitrary positions, DeepSeek-style FP8 block-scaled GEMM. Every kernel has a PyTorch reference path and a parity test. |
| Rewards | `kiln/reward` | Composable rule verifiers (`\boxed{}` math with fraction/sympy equivalence, format, repetition, length, sandboxed Python unit tests), FastAPI reward server with Prometheus metrics, retrying batched client, reward-model scorer. |
| Checkpointing | `kiln/checkpoint` | Async, atomic, rank-sharded; safetensors for weights; back-pressure; writer errors surfaced, never swallowed. |
| Serving | `kiln/serving` | Prefix-cache-aware router (block radix tree + load cap), continuous-batching scheduler with paged KV, chunked prefill and youngest-first preemption (doubles as a capacity-planning simulator), SLO-gated canary controller, OpenAI-compatible gateway. |
| Ops | `configs/`, `deploy/`, `.github/` | Strict pydantic YAML schema with env expansion and CLI overrides; KubeRay job, reward server with no-egress NetworkPolicy, gateway deployment; Slurm scripts; CI with lint, CPU tests on two Pythons, perf-regression gate, GPU runner job. |

## Design notes

The four ADRs in [`docs/design/`](docs/design) explain the decisions that are
not obvious from the code:

1. [Asynchronous training with bounded staleness](docs/design/01-async-training.md) — why a bounded buffer, how importance weighting keeps it correct, how depth adapts.
2. [Fused kernels](docs/design/02-kernels.md) — the online-softmax log-prob/entropy kernel and its one-pass backward, FP8 block scaling, why not Liger / `torch.compile`.
3. [Weight sync and checkpointing](docs/design/03-weight-sync-and-checkpointing.md) — NCCL broadcast into live engine workers, pinned-buffer async checkpoints with atomic publish.
4. [Serving control plane](docs/design/04-serving.md) — cache-aware routing, scheduler simulator, canary guardrails.

## Running things

```bash
# validate every shipped config against the strict schema
kiln validate configs/*.yaml

# override anything from the CLI
kiln train configs/grpo_math_qwen3_8b.yaml --set rollout.sampling.n=16 --set async.max_staleness=3

# reward server (CPU, runs model-written code -> deploy with deploy/k8s/reward-server.yaml, which has no egress)
kiln reward-serve configs/grpo_math_qwen3_8b.yaml --port 8008

# inference gateway with a canary ramped by SLO
kiln gateway --replica base-0=http://vllm-0:8000 --replica canary-0=http://vllm-1:8000 --canary canary-0 --p99-slo-s 0.5

# benchmarks measure *this* machine and print what they measured
kiln bench kernels | step | scheduler | sync | all

# topology discovery -> recommended NCCL env for the fabric actually present
kiln topo
```

`examples/` has three short scripts: the async loop on CPU with a learning
curve, custom reward rules and engines, and replica sizing from the
scheduler simulator.

## Testing philosophy

* **Alignment is where RL code goes wrong.** The `[B, T]` / `[B, T-1]`
  convention, the behaviour-log-prob offset, packing/unpacking and the
  log-prob kernel each have unit tests, and `test_e2e.py` runs the whole
  loop on a synthetic dense reward: if any link in the chain is off by one,
  the reward curve does not go up and the test fails.
* **Async must be checkable against sync.** With `max_staleness=0` the run
  is bit-for-bit deterministic and every importance weight is exactly 1.
* **Kernels are tested forward and backward** at fp32 and bf16, at
  non-power-of-two sizes, with extreme logits, and (on CUDA) Triton against
  the reference path.
* **Infrastructure gets failure tests**: checkpoint writer errors surface,
  live weights can be mutated while a write is in flight, a dead rollout
  worker fails the run instead of hanging it, a tripped canary stays tripped.

## Status and honesty

This is infrastructure code, tested on CPU in CI and exercised on GPU
hardware I have access to. The throughput numbers you get from `kiln bench`
are the throughput numbers on *your* hardware; nothing in this repo claims
a number it did not measure in the run that printed it. The vLLM/SGLang
engines track upstream APIs that move quickly — pin the versions in
`deploy/docker/Dockerfile` if you are deploying.

Roadmap: pipeline-parallel schedule inside the trainer (the mesh and groups
exist, the 1F1B schedule does not yet), MoE expert parallelism, TensorRT-LLM
rollout engine, GSPO as a first-class algorithm.

## License

Apache-2.0. Author: Priyanshu Singh (ps2001@bu.edu).
