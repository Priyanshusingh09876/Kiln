"""Data-parallel FSDP2 validation run, any world size, CPU (gloo) or GPU (NCCL).

CI runs it as ``torchrun --nproc_per_node=2`` from tests/test_multirank.py on CPU. On GPUs
(see docs/VALIDATION.md) the same file runs unchanged under e.g.
``torchrun --nproc_per_node=8`` or ``torchrun --nnodes=2 --nproc_per_node=8 ...``; the
world size becomes ``parallel.dp``.

Checks, with real process groups and real FSDP2 sharding:

1. parameters are actually sharded (DTensor) and ranks see *different* prompts;
2. after training, the gathered full weights are identical on both ranks
   (gradients are being all-reduced, not silently trained per-rank);
3. a run interrupted at step 2 and resumed from its rank-sharded checkpoint
   ends bit-identical to an uninterrupted 4-step run (weights, losses), and
   the checkpoint directory holds one shard per rank.

Rank 0 prints one JSON line prefixed ``RESULT`` for the test to parse.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from kiln.config.schema import KilnConfig
from kiln.distributed import full_state_dict
from kiln.orchestration import AsyncController

ROWS = [{"prompt": f"prompt number {i}"} for i in range(32)]


def reward_fn(items):
    r = [max(0.0, 1.0 - abs(ord(it["response"][:1] or "\0") - ord("7")) / 128) for it in items]
    return r, [{"r": v} for v in r]


def cfg(root: Path, ckpt: str, run: str, total: int) -> KilnConfig:
    return KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 2, "tiny_heads": 2},
            "parallel": {"dp": int(os.environ.get("WORLD_SIZE", "2"))},
            "train": {
                "rollout_batch_size": 16,
                "micro_batch_size": 8,
                "total_steps": total,
                "optim": {"lr": 1e-2, "warmup_steps": 1},
            },
            "rollout": {"num_engines": 1, "sampling": {"n": 4, "max_new_tokens": 4}},
            "algorithm": {"name": "grpo", "kl": {"coef": 0.01}},
            "async": {"enabled": False, "queue_depth": 1, "max_staleness": 0},
            "checkpoint": {"dir": str(root / ckpt), "interval": 2, "async_save": True},
            "logging": {
                "log_dir": str(root / "runs"),
                "run_name": f"{run}_r{dist.get_rank() if dist.is_initialized() else 0}",
            },
        }
    )


def gathered(ctl) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in full_state_dict(ctl.trainer.model).items()}


def main() -> int:
    root = Path(os.environ.get("KILN_MR_DIR") or tempfile.mkdtemp())
    out: dict = {}

    # ---- uninterrupted 4 steps
    probe = AsyncController.from_config(cfg(root, "ck_probe", "probe", 4), rows=ROWS, reward_fn=reward_fn)
    out["sharded"] = type(next(probe.trainer.model.parameters())).__name__ == "DTensor"
    first_prompts = [s.prompt_text for s in probe.workers[0].produce_batch()[0].samples]
    probe.shutdown()

    a = AsyncController.from_config(cfg(root, "ck_a", "a", 4), rows=ROWS, reward_fn=reward_fn)
    ra = a.run()
    wa = gathered(a)

    # ranks saw different prompts?
    all_prompts = [None] * dist.get_world_size()
    dist.all_gather_object(all_prompts, sorted(set(first_prompts)))
    out["disjoint_prompts"] = all(
        set(all_prompts[i]).isdisjoint(all_prompts[j])
        for i in range(len(all_prompts))
        for j in range(i + 1, len(all_prompts))
    )

    # full weights identical across ranks?
    sums = torch.stack([v.double().sum() for v in wa.values()]).to(a.trainer.device)
    gathered_sums = [torch.zeros_like(sums) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_sums, sums)
    out["ranks_in_sync"] = all(bool(torch.equal(gathered_sums[0], g)) for g in gathered_sums[1:])
    out["world_size"] = dist.get_world_size()
    out["device"] = str(a.trainer.device)
    out["backend"] = dist.get_backend()

    # ---- interrupted at 2, resumed to 4
    b1 = AsyncController.from_config(cfg(root, "ck_b", "b1", 4), rows=ROWS, reward_fn=reward_fn)
    b1.run(max_steps=2)
    dist.barrier()
    b2 = AsyncController.from_config(cfg(root, "ck_b", "b2", 4), rows=ROWS, reward_fn=reward_fn)
    rb = b2.run()
    wb = gathered(b2)
    out["resume_steps"] = rb.steps
    out["resume_weights_equal"] = all(torch.equal(wa[k], wb[k]) for k in wa)
    out["resume_losses_equal"] = ra.series("loss/total")[2:] == rb.series("loss/total")
    shard_files = sorted(p.name for p in (root / "ck_b" / "step_2").iterdir()) if dist.get_rank() == 0 else []
    out["shard_files"] = shard_files
    out["final_reward"] = ra.series("reward/mean")[-1]

    flags = [
        out["sharded"],
        out["disjoint_prompts"],
        out["ranks_in_sync"],
        out["resume_weights_equal"],
        out["resume_losses_equal"],
    ]
    if dist.get_rank() == 0:
        print("RESULT " + json.dumps(out), flush=True)
    dist.barrier()
    return 0 if all(flags) else 1


if __name__ == "__main__":
    rc = main()
    dist.destroy_process_group()
    sys.exit(rc)
