# Docs

* [`design/01-async-training.md`](design/01-async-training.md) — bounded-staleness async loop, importance weighting, adaptive depth
* [`design/02-kernels.md`](design/02-kernels.md) — fused Triton kernels and their reference paths
* [`design/03-weight-sync-and-checkpointing.md`](design/03-weight-sync-and-checkpointing.md) — NCCL weight sync into live engines, async atomic checkpoints
* [`design/04-serving.md`](design/04-serving.md) — cache-aware routing, scheduler simulator, SLO-gated canaries
* [`metrics.md`](metrics.md) — every metric the trainer exports and what to do when it moves
