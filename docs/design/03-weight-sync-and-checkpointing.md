# ADR-03: Weight synchronisation and asynchronous checkpointing

**Status:** accepted · **Modules:** `kiln.rollout.weight_sync`, `kiln.checkpoint`

Both are instances of the same principle: *the GPU should only ever wait for
a device-to-host (or device-to-device) copy, never for serialisation, a
filesystem, or a network round trip.*

## Weight sync (trainer → rollout engines)

| transport | path | when |
|---|---|---|
| `inproc` | hand tensors over | mock engine, tests |
| `cpu` | gather on rank 0 → CPU → Ray object store → engine `load_weights` | < 10 B params or infrequent sync |
| `nccl` | dedicated NCCL group {trainer rank 0} ∪ {every engine TP worker}; broadcast in ~1 GB buckets | production |

Details that matter:

* **FSDP2 DTensors** must be gathered before sending (`iter_sync_tensors`
  handles `full_tensor()`), and wrapper prefixes
  (`_fsdp_wrapped_module.`, `_checkpoint_wrapped_module.`, `module.`)
  are stripped so names match what vLLM's `load_weights` expects.
* **Bucketing** (`distributed.comm.plan_buckets`) keeps the number of
  collectives small; a 70B model in bf16 is ~140 buckets instead of ~700
  tensors.
* **vLLM side**: `KilnWorkerExtension` runs *inside* each TP worker. It
  joins the sync group, receives the broadcast, and calls
  `model_runner.model.load_weights` in place — the KV cache and CUDA graphs
  survive the update, so there is no warm-up penalty after each step.
* **LoRA-only sync** is a `filter_fn`; the transfer drops from tens of GB to
  tens of MB.
* **Partial rollouts**: engines are `pause()`d at a token boundary around
  the swap and `resume()`d after. Samples record the version they *started*
  under (conservative for staleness accounting).

## Checkpointing

Synchronous `torch.save` of a sharded model + Adam state stalls every rank
for serialisation + write time. The cost splits into a fast, unavoidable
device→host copy and a slow, GPU-irrelevant write. `AsyncCheckpointer`:

1. Copies each tensor into a **pinned host buffer** (allocated once, reused
   every checkpoint) with `non_blocking=True`, then `cuda.synchronize()`.
   This is the only part training waits for.
2. Hands the host buffers to a background thread that writes
   `rank<i>.safetensors` (weights) + `rank<i>.pt` (optimizer / scheduler /
   step, structure preserved) under `step_<N>.tmp/`.
3. **Atomic publish**: barrier across ranks → rank 0 writes `meta.json` →
   `os.replace(tmp, final)` → `LATEST` pointer updated last via another
   `os.replace`. A crash at any point leaves either a complete checkpoint or
   a `.tmp` directory that is ignored on resume.
4. **Back-pressure**: a second `save()` while a write is in flight blocks
   rather than allocating more host memory. A failure in the writer thread
   is re-raised on the next `save()`/`wait()`, never swallowed.

`tests/test_checkpoint_and_config.py::test_training_can_mutate_source_tensors_while_write_in_flight`
is the test that matters: it mutates the live weights right after `save()`
returns and verifies the checkpoint still holds the old values.

## Multi-rank details learned the hard way

* The writer thread coordinates ranks with barriers. Those must go through a
  **dedicated gloo process group** created on the main thread, never the
  training group: two threads issuing collectives on one group can interleave
  and deadlock.
* Each rank saves its local FSDP2 shards as plain tensors. Loading copies them
  back into `param.to_local()` (`load_sharded_state_dict`), and optimizer state
  is re-wrapped as DTensors matching each parameter (`rewrap_optimizer_state`),
  otherwise the first post-resume step mixes DTensor parameters with plain state.
* Rollout engines hold their own unsharded copy of the policy. Giving them the
  trainer's sharded module would make generation issue FSDP collectives on
  different ranks at different times.

`tests/test_multirank.py` runs two real processes over gloo and checks sharding,
gradient sync, rank-sharded checkpoints and a bit-exact resume.
