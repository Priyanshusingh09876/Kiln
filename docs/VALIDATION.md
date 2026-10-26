# GPU and multi-node validation plan

Everything marked 🟡 in `docs/STATUS.md` needs hardware this repository has not run
on yet. This page lists, in order of cost, exactly what to run, what to record,
and where the results go. A tier counts as done only when its outputs are committed
to `docs/benchmarks/` with the command, hardware and software versions that
produced them.

Rough cost at on-demand prices: tier 1 ≈ $5, tier 2 ≈ $40–60, tier 3 ≈ $100+.

## Tier 0 — any laptop (done in CI on every commit)

```bash
pip install torch && pip install --no-build-isolation -e ".[dev]"
pytest -q                                              # includes the 2-process FSDP2 test
python scripts/bench_kernels.py --tag cpu              # reference-path correctness
python scripts/bench_sync_vs_async.py --tag cpu-sim    # scheduling simulation, ~7 min
python examples/04_gateway_canary_demo.py              # routing + canary under load, ~3 min
```

## Tier 1 — one GPU (kernels)

Any sm80+ GPU; sm90 (H100/H200) also enables the FP8 GEMM path.

```bash
# CUDA toolkit with nvcc must be on PATH so kiln._cuda gets built
nvcc --version && nvidia-smi
pip install torch triton && pip install --no-build-isolation -e ".[dev]"
python -c "from kiln import native; print(native.summary())"     # expect cuda_ext: True, backend: cuda

pytest -q tests/test_kernels.py tests/test_native.py              # parity: Triton and CUDA C++ vs reference
python scripts/bench_kernels.py --tag <gpu-name>                  # time, max error, peak memory, all backends
```

Record: `docs/benchmarks/kernels_<gpu-name>.{md,json}` (written by the script), plus
the `pytest` summary line and `nvidia-smi` header in the PR description.

What would count as a failure worth fixing before anything else: any ❌ in the
"within tolerance" column; the fused log-prob kernel using *more* peak memory than
`naive` at 128K vocab (the whole point of it is the opposite); the CUDA C++ path
failing to build.

## Tier 2 — one node, 8 GPUs (distributed training, rollout engines)

### 2a. FSDP2 over NCCL — runs today, no code changes

```bash
KILN_MR_DIR=/tmp/kiln-mr torchrun --nproc_per_node=8 --standalone tests/multirank/run_dp2.py
```

Same checks as the CPU test, now with NCCL and CUDA tensors: parameters sharded,
ranks see disjoint prompts, gathered weights identical across ranks, and a run
interrupted at step 2 resumes bit-identical to an uninterrupted one. Expect
`"backend": "nccl"` in the RESULT line. Record the RESULT line and wall time.

### 2b. Async vs sync with real vLLM engines — needs work first

The benchmark harness (`scripts/bench_sync_vs_async.py --config ...`) drives the
in-process controller. With real vLLM engines that means trainer and engines would
share GPUs without placement, so the meaningful GPU comparison has to go through
the Ray runtime (`kiln train ... --ray`), which has never been run. Expected order
of work:

1. `pip install -e ".[ray,vllm,train]"`; run `kiln train configs/grpo_math_qwen3_8b.yaml --ray --steps 5`
   with a small model (e.g. Qwen2.5-0.5B), fix what breaks, add a test.
2. Add a `--ray` mode to `scripts/bench_sync_vs_async.py`.
3. Run sync vs async on the same 8 GPUs (e.g. 4 trainer + 4 engine), same seeds, ≥ 200 steps
   of a real math dataset; record wall time, trainer busy fraction, tokens/s, time to a
   reward threshold, final reward, KL, staleness, IS weights, GPU utilisation (the script
   samples NVML automatically) to `docs/benchmarks/sync_vs_async_<tag>.{md,json}`.

### 2c. NCCL weight sync into live vLLM workers

Time `WeightSyncer(transport="nccl")` for a 7–8B model, compare with the CPU transport,
and confirm generation after a sync uses the new weights (log-probs from the engine
should match the trainer's within bf16 tolerance). Record GB/s and seconds per sync.

## Tier 3 — two or more nodes (multi-node)

### 3a. Cross-node FSDP2 + checkpoint/resume

Shared filesystem for `KILN_MR_DIR` is required (checkpoints are rank-sharded).

```bash
# on each node, NODE_RANK = 0 / 1, MASTER = first node's address
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
         --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 tests/multirank/run_dp2.py
```

Before running: `kiln topo` on each node and record the recommended NCCL environment.
Record: topology (GPUs, NVLink/NVSwitch, NICs), NCCL version, RESULT line, wall time.

### 3b. Failure recovery

1. Start a 2-node training run with `checkpoint.interval` small (e.g. 10).
2. Kill one rank's process mid-run (`kill -9`).
3. Expected: the surviving ranks fail within the NCCL timeout instead of hanging
   forever (record how long it takes), no partially-written checkpoint directory is
   ever reported as LATEST, and relaunching the same command resumes from the last
   complete checkpoint.
4. For a synchronous-mode run, compare the resumed run's loss curve with an
   uninterrupted run: it should match exactly, as it does on CPU.

Record the timeline (kill time, failure detected, relaunch, first post-resume step)
and the loss-curve comparison.

## Results log

| Tier | Date | Hardware | Result | Link |
|---|---|---|---|---|
| 0 | 2026-10-26 | CPU container | all passing | `docs/benchmarks/*cpu*` |
| 1 (partial) | 2026-10-26 | RTX 3080 Ti Laptop 16 GB, Windows, torch 2.14 + CUDA 12.6, triton-windows | full suite passes after fixing 5 bugs found on this machine (STATUS.md #16–20); Triton kernels benchmarked; CUDA C++ ext not built | `docs/benchmarks/*rtx3080-laptop*` |
| 1 (partial) | 2026-10-26 | same | fused log-prob v2 + baselines: 2.1× over v1, 4.0× over `torch.compile`, 73–86% of copy BW at 4096×128k; RMSNorm/RoPE slower than `torch.compile` | `kernels_rtx3080-laptop-v2.md`, `tune_logprob_rtx3080-laptop.md` |
| 2a | — | — | not run yet | |
| 2b | — | — | needs Ray path first | |
| 2c | — | — | not run yet | |
| 3a | — | — | not run yet | |
| 3b | — | — | not run yet | |
