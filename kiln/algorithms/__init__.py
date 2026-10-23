from kiln.algorithms.base import Algorithm, ForwardOutputs
from kiln.algorithms.dpo import DPO
from kiln.algorithms.grpo import GRPO
from kiln.algorithms.ppo import PPO
from kiln.config.schema import AlgorithmConfig, AsyncConfig


def build_algorithm(cfg: AlgorithmConfig, async_cfg: AsyncConfig) -> Algorithm:
    if cfg.name in ("grpo", "dr_grpo", "rloo", "reinforce_pp"):
        return GRPO(cfg, async_cfg)
    if cfg.name == "ppo":
        return PPO(cfg, async_cfg)
    if cfg.name == "dpo":
        return DPO(cfg, async_cfg)
    raise ValueError(f"unknown algorithm {cfg.name!r}")


__all__ = ["DPO", "GRPO", "PPO", "Algorithm", "ForwardOutputs", "build_algorithm"]
