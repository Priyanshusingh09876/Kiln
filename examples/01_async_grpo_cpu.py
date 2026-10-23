"""Run async GRPO end-to-end on CPU in ~15 seconds and plot the learning curve.

    python examples/01_async_grpo_cpu.py

The reward is dense and synthetic ("make the first byte of the response close
to '7'") so you can watch GRPO work without a GPU, a dataset or a reward
model. Swap `reward_fn` for anything with the same signature.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kiln.config.schema import KilnConfig
from kiln.orchestration import AsyncController

TARGET = ord("7")


def reward_fn(items):
    rewards = []
    for it in items:
        c = it["response"][:1]
        rewards.append(max(0.0, 1.0 - abs(ord(c) - TARGET) / 128) if c else 0.0)
    return rewards, [{"dist": r} for r in rewards]


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kiln-example-"))
    cfg = KilnConfig.model_validate(
        {
            "model": {"dtype": "fp32", "tiny_hidden": 32, "tiny_layers": 1, "tiny_heads": 2},
            "train": {"rollout_batch_size": 32, "micro_batch_size": 16, "total_steps": 60, "optim": {"lr": 1e-2, "warmup_steps": 2}},
            "rollout": {"num_engines": 2, "sampling": {"n": 8, "max_new_tokens": 4}},
            "algorithm": {"name": "grpo", "kl": {"coef": 0.0}},
            "async": {"queue_depth": 2, "max_staleness": 2},
            "checkpoint": {"dir": str(tmp / "ckpt"), "interval": 30},
            "logging": {"log_dir": str(tmp / "runs"), "run_name": "example", "backend": "stdout"},
        }
    )
    rows = [{"prompt": f"prompt {i}"} for i in range(16)]
    ctl = AsyncController.from_config(cfg, rows=rows, reward_fn=reward_fn)
    res = ctl.run()
    r = res.series("reward/mean")
    print("\nreward every 10 steps:", [round(sum(r[i : i + 10]) / 10, 3) for i in range(0, len(r), 10)])
    print("mean staleness:", round(sum(res.series("async/mean_staleness")) / len(r), 2))
    print("metrics:", res.run_dir / "metrics.jsonl")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].plot(r); ax[0].set_title("reward/mean"); ax[0].set_xlabel("step")
        ax[1].plot(res.series("async/is_weight")); ax[1].set_title("mean IS weight (staleness correction)"); ax[1].set_xlabel("step")
        fig.tight_layout(); fig.savefig(tmp / "curve.png", dpi=120)
        print("plot:", tmp / "curve.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
