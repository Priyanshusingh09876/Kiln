"""The rollout-engine contract.

Every inference backend (vLLM, SGLang, the in-process mock) implements this
small protocol so the orchestrator never has to know which one it is talking
to. Two calls matter for asynchronous training and deserve explanation:

``update_weights(named_tensors, version)``
    Receive new policy weights and tag all *subsequent* generations with
    ``version``. Implementations may receive tensors over NCCL, CUDA IPC, or
    HTTP depending on where they live relative to the trainer.

``pause()`` / ``resume()``
    For *partial rollouts*: freeze in-flight requests at a token boundary,
    swap weights, and continue generating. Samples produced this way have
    tokens from two policy versions; we record the version at which each
    sample *started* (the conservative choice for staleness accounting) and
    the engine's reported log-probs remain the correct behaviour policy for
    importance weighting regardless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from kiln.config.schema import SamplingConfig
from kiln.rollout.types import Sample


@dataclass
class GenerationRequest:
    prompt_ids: list[int]
    group_id: int
    prompt_text: str = ""
    n: int = 1
    meta: dict | None = None


class RolloutEngine(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.policy_version: int = 0
        self._paused = False

    @abstractmethod
    def generate(self, requests: list[GenerationRequest], sampling: SamplingConfig) -> list[Sample]:
        """Return ``sum(r.n for r in requests)`` samples, each carrying ``behavior_logprobs``."""

    @abstractmethod
    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None: ...

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self) -> None:  # noqa: B027 - optional hook
        """Release KV cache / weights to free GPU memory (colocated mode)."""

    def wake(self) -> None:  # noqa: B027 - optional hook
        """Reload after :meth:`sleep`."""

    def shutdown(self) -> None:  # noqa: B027 - optional hook
        """Release all resources."""

    def stats(self) -> dict[str, float]:
        return {}
