# First GPU bring-up: 22 failures, 5 bugs, and what CPU CI could not see

**Date:** 2026-10-26 · **Hardware:** one NVIDIA RTX 3080 Ti Laptop GPU (16 GB, Ampere sm_86), Windows 11 (WDDM),
torch 2.14.1 + CUDA 12.6, `triton-windows` · **Starting point:** every test passing on Linux CPU.

Kiln was developed against CPU CI: GPU tests skip when there is no CUDA device, and the Triton kernels had
only ever been checked through their PyTorch reference paths. This is the record of the first run on a real
GPU — what broke, why CPU CI missed each one, and the guard added so it cannot recur.

## Timeline

| run | result | root causes |
|---|---|---|
| 1 | 22 failed | bugs 1–3 below; one kernel bug cascaded into most of the failures |
| 2 | 5 failed | bug 4, plus one test that assumed no GPU exists |
| 3 | 1 failed | a test compared a CPU tensor with a CUDA tensor |
| 4 | all pass | — |
| benchmarks | crashed writing the report after a 10-minute run | bug 5 |

## The bugs

### 1. Fused log-prob kernel returned `+inf` for every token

**Symptom.** All Triton log-prob parity tests failed with `logp = +inf`. RMSNorm and RoPE kernels passed, so the
toolchain was fine. Downstream, the loss was infinite, the first optimizer step turned the weights into NaN, and
`torch.multinomial` in the rollout engine raised on the NaN probabilities — which is why every end-to-end,
resume and DPO test failed too. Nineteen of the 22 failures were this one line.

**Cause.** A formatting edit had attached a comment to the online-softmax update and swallowed half the expression:

```python
l = l * scale  # noqa: E741 - online-softmax denominator, matches the paper + tl.sum(e, axis=0)
```

`l` (the running sum of `exp(x - m)`) therefore stayed 0, `lse = m + log(0) = -inf`, and `logp = x[t] - lse = +inf`.

**Why CI missed it.** The kernel body had never executed: CPU CI only ran the PyTorch reference, and the
GPU tests skipped.

**Fix and guard.** Restore the term. Add `tests/test_triton_interpret.py`, which runs the real Triton forward and
backward kernels on CPU under `TRITON_INTERPRET=1` and compares them with a float64 autograd reference, at a
vocabulary larger than the 8192 block cap so the multi-block online-softmax path is exercised. The test fails on
the old code. Also: `generate()` now raises a clear `FloatingPointError` on non-finite logits instead of a
confusing error deep inside `multinomial`.

### 2. A metric created on CPU broke the cross-rank reduction

`kl_term` returned `"kl/coef": torch.tensor(coef)` — a CPU tensor — next to CUDA losses, and
`all_reduce_metrics` stacks them into one tensor. Invisible on CPU, where every tensor is on the same device.
Fixed at the source, and `all_reduce_metrics` now moves mixed-device metrics onto the accelerator.

### 3. The PPO critic depended on an SVD that does not converge on CUDA

The critic recovered hidden states as `logits @ pinverse(lm_head.weight)` — an approximation that happened to
work on CPU. cuSOLVER's SVD did not converge. Replaced with `TinyLM.forward_hidden`, which returns the real final
hidden states and goes through `__call__` so FSDP2's pre-forward hooks still unshard the root parameters.

### 4. Resume moved the sampler's RNG state onto the GPU

Checkpoints are loaded with `map_location=trainer.device`. That also moved the rollout engine's CPU
`torch.Generator` state onto CUDA, and `Generator.set_state` only accepts a CPU `ByteTensor`. Only reachable with a
CUDA trainer. The engine now moves the state back to CPU on load.

### 5. Windows wrote the benchmark report in cp1252

`bench_sync_vs_async.py` finished all nine runs and then crashed writing a Markdown report containing `≥`:
`Path.write_text` without an encoding uses the platform default, which is cp1252 on Windows. The raw JSON had been
written first, so the report was regenerated with `--from-json` instead of rerunning. Every text read and write in
the repo now names UTF-8, and `tests/test_text_encoding.py` parses the source to keep it that way — Linux CI, where
the default is already UTF-8, would never catch a regression otherwise.

Two failures were test bugs rather than product bugs: one asserted GPU utilisation is always "n/a" (true only
without a GPU), and one called `torch.equal` on the engine's CPU weights and the trainer's CUDA weights.

## Results after the fixes

Full numbers, including fp32 and error columns: [`benchmarks/kernels_rtx3080-laptop.md`](benchmarks/kernels_rtx3080-laptop.md).
Forward + backward, median of 20 iterations, every row within tolerance of a float64 reference.

| kernel (bf16) | shape | naive PyTorch | Triton | speedup | peak memory |
|---|---|---|---|---|---|
| log-prob + entropy | 4096 × 128k vocab | 153.6 ms | 20.8 ms | 7.4× | 10.0 GB → 3.0 GB |
| log-prob + entropy | 2048 × 152k vocab | 91.2 ms | 15.2 ms | 6.0× | 5.9 GB → 1.8 GB |
| RMSNorm | 16384 × 4096 | 32.5 ms | 4.6 ms | 7.0× | 1.5 GB → 0.56 GB |
| RoPE | 4 × 4096 × 32 × 128 | 19.0 ms | 5.5 ms | 3.4× | 0.52 GB → 0.26 GB |

The memory saving is what matters for RL training: the naive path materialises several `[tokens, vocab]` fp32
tensors, and at a 128k vocabulary that, not compute, limits the micro-batch size.

The async-vs-sync scheduling benchmark reproduced on this machine
([`benchmarks/sync_vs_async_rtx3080-laptop.md`](benchmarks/sync_vs_async_rtx3080-laptop.md)): async with queue
depth 4 reached the target reward 2.0× sooner in wall time, with the trainer busy 54% of the time instead of 5%.
That benchmark simulates generation latency, so it measures scheduling, not GPU throughput.

## What the benchmark exposed, and what changed after it

Kiln's own PyTorch fallback for log-prob + entropy (`KILN_KERNEL_BACKEND=torch`, used only where Triton is
unavailable) is correct but slower than the naive formula, and at 4096 × 128k it peaked at 16 GB — the whole card —
taking 1.4 s. It saves full-size `[N, V]` fp32 intermediates for backward. The fix is to process tokens in chunks
and recompute in backward, as the Triton kernel does. **Now done**: the fallback is chunked over rows and writes the
gradient in the input dtype; the GPU re-measurement is pending.

Reading the numbers more carefully also showed that "7.4× faster than naive" is the wrong yardstick. At 4096 × 128k
in bf16 the kernel moves at least 3 GB (read the logits twice, write the gradient), so 20.8 ms is about 150 GB/s —
roughly a third of what this GPU can copy. Two causes were visible in the code: the backward wrote an fp32 gradient
and converted it to bf16 in a separate pass (an extra 2 GB buffer and two extra passes over memory), and the block
size and warp count were fixed by hand. Version 2 stores the gradient directly in the logits dtype, autotunes block
size and warps, runs the backward on a 2-D grid of independent tiles, and splits each row across several programs
when the batch is too short to fill the GPU. `scripts/tune_fused_logprob.py` measures every configuration against
v1, and `scripts/bench_kernels.py` now compares against `torch.compile` and Liger Kernel and reports % of measured
copy bandwidth. Results go here once they are measured.

## Lessons

1. **A skipped test is not a passing test.** The worst bug was in code that had never executed. Where hardware
   is unavailable, run the code another way — here, Triton's interpreter — rather than only testing a reference.
2. **Device placement is an invariant, not an implementation detail.** Three of the five bugs were a tensor on
   the wrong device, and all three were invisible on a single-device CPU run.
3. **Write raw results before formatting them.** Saving JSON before rendering the report turned a crash at
   minute ten into a ten-second regeneration.
4. **Check portability in the source when CI cannot reproduce the platform.** The encoding test runs on Linux
   but protects Windows users.

## Reproduce

```bash
pip install -e ".[dev]"            # CUDA build of torch; on Windows also: pip install triton-windows
pytest -q
python scripts/bench_kernels.py --tag <your-gpu>
python scripts/bench_sync_vs_async.py --tag <your-gpu> --seeds 0 1 2
```
