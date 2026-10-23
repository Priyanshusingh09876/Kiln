"""Plugging in your own pieces: a custom reward rule and a custom rollout engine.

Shows the two extension points most teams need first:

1. ``@rules.register("my_rule")`` — a verifier composable from YAML with weights.
2. Subclass :class:`RolloutEngine` — e.g. to call a proprietary inference service.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from kiln.config.schema import RewardConfig, RewardFnConfig, SamplingConfig
from kiln.reward import RewardScorer, rules
from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.types import Sample


@rules.register("contains_keyword")
def contains_keyword(response: str, answer: str | None, keyword: str = "because", **_) -> float:
    return 1.0 if keyword in response.lower() else 0.0


class EchoEngine(RolloutEngine):
    """Toy engine that 'generates' the prompt reversed. Replace ``generate`` with an RPC to your service."""

    name = "echo"

    def generate(self, requests: list[GenerationRequest], sampling: SamplingConfig) -> list[Sample]:
        out = []
        for r in requests:
            for _ in range(r.n or sampling.n):
                ids = list(reversed(r.prompt_ids))[: sampling.max_new_tokens]
                out.append(Sample(r.prompt_ids, ids, [-1.0] * len(ids), r.group_id, self.policy_version, meta=dict(r.meta or {})))
        return out

    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        # ship `named_tensors` to the service here
        self.policy_version = version


if __name__ == "__main__":
    scorer = RewardScorer(RewardConfig(functions=[RewardFnConfig(name="contains_keyword", params={"keyword": "therefore"}),
                                                  RewardFnConfig(name="math_boxed", weight=2.0)]))
    print(scorer.score_batch([{"response": r"therefore \boxed{4}", "answer": "4"}, {"response": "no", "answer": "4"}]))
    eng = EchoEngine()
    print(eng.generate([GenerationRequest([1, 2, 3], group_id=0, n=2)], SamplingConfig(max_new_tokens=8))[0])
