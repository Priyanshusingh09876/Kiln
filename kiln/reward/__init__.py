"""Reward computation: rule verifiers, reward models, and the HTTP server/client."""

from __future__ import annotations

from kiln.config.schema import RewardConfig
from kiln.reward import rules
from kiln.reward.rules import answers_match, extract_boxed, extract_final_answer, normalize_answer
from kiln.reward.server import RewardClient, RewardScorer, create_app


def build_reward_fn(cfg: RewardConfig):
    """Return ``fn(items) -> (rewards, breakdown)`` for the configured mode."""
    if cfg.mode == "server":
        assert cfg.server_url, "reward.mode=server requires reward.server_url"
        client = RewardClient(cfg.server_url, cfg.server_timeout_s, batch_size=cfg.batch_size)
        return client.score
    scorer = RewardScorer(cfg)
    return scorer.score_batch


__all__ = [
    "RewardClient",
    "RewardScorer",
    "answers_match",
    "build_reward_fn",
    "create_app",
    "extract_boxed",
    "extract_final_answer",
    "normalize_answer",
    "rules",
]
