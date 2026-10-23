"""vLLM rollout engine.

Runs a ``vllm.LLM`` instance (optionally tensor-parallel) and exposes
Kiln's :class:`RolloutEngine` contract. Weight updates go through
``collective_rpc`` into a worker extension that does an in-place
``load_weights`` on the model runner, so the KV cache and CUDA graphs stay
warm across policy versions — the pattern verl/OpenRLHF converged on. Weights
arrive either as CPU tensors (simple, slow) or via a NCCL process group that
the trainer broadcasts into (``init_weight_sync_group``).

The import is deferred so this module can be imported (and the rest of the
test-suite can run) on machines without vLLM.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import torch

from kiln.config.schema import RolloutConfig, SamplingConfig
from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.types import Sample


class KilnWorkerExtension:
    """Mixed into vLLM's GPU worker via ``worker_extension_cls``.

    Methods here execute *inside* every TP worker process.
    """

    def kiln_init_sync_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int
    ) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        rank = get_world_group().rank + rank_offset
        pg = StatelessProcessGroup.create(
            host=master_address, port=master_port, rank=rank, world_size=world_size
        )
        self._kiln_pg = PyNcclCommunicator(pg, device=self.device)  # type: ignore[attr-defined]

    def kiln_recv_weight(self, name: str, dtype_str: str, shape: list[int]) -> None:
        dtype = getattr(torch, dtype_str.split(".")[-1])
        buf = torch.empty(shape, dtype=dtype, device=self.device)  # type: ignore[attr-defined]
        self._kiln_pg.broadcast(buf, src=0)
        self.model_runner.model.load_weights(weights=[(name, buf)])  # type: ignore[attr-defined]
        del buf

    def kiln_load_cpu_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        self.model_runner.model.load_weights(weights=[(n, t.to(self.device)) for n, t in weights])  # type: ignore[attr-defined]


class VLLMEngine(RolloutEngine):
    name = "vllm"

    def __init__(self, model_path: str, cfg: RolloutConfig, seed: int = 0, dtype: str = "bfloat16"):
        super().__init__()
        try:
            from vllm import LLM
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'kiln-rl[vllm]'") from e
        self.cfg = cfg
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_prefix_caching=cfg.enable_prefix_caching,
            dtype=dtype,
            seed=seed,
            enable_sleep_mode=True,
            worker_extension_cls="kiln.rollout.vllm_engine.KilnWorkerExtension",
        )
        self._tokens = 0
        self._seconds = 0.0

    def _sampling_params(self, s: SamplingConfig, n: int):
        from vllm import SamplingParams

        return SamplingParams(
            n=n,
            temperature=s.temperature,
            top_p=s.top_p,
            top_k=s.top_k,
            max_tokens=s.max_new_tokens,
            stop=s.stop or None,
            logprobs=0,  # sampled-token logprob only
        )

    def generate(self, requests: list[GenerationRequest], sampling: SamplingConfig) -> list[Sample]:
        t0 = time.perf_counter()
        version = self.policy_version
        prompts = [{"prompt_token_ids": r.prompt_ids} for r in requests]
        n = requests[0].n or sampling.n
        outputs = self.llm.generate(prompts, self._sampling_params(sampling, n), use_tqdm=False)
        samples: list[Sample] = []
        for req, out in zip(requests, outputs):
            for comp in out.outputs:
                ids = list(comp.token_ids)
                lps = (
                    [next(iter(lp.values())).logprob for lp in comp.logprobs]
                    if comp.logprobs
                    else [0.0] * len(ids)
                )
                self._tokens += len(ids)
                samples.append(
                    Sample(
                        prompt_ids=list(req.prompt_ids),
                        response_ids=ids,
                        behavior_logprobs=lps,
                        group_id=req.group_id,
                        policy_version=version,
                        prompt_text=req.prompt_text,
                        response_text=comp.text,
                        finish_reason=comp.finish_reason or "stop",
                        meta=dict(req.meta or {}),
                    )
                )
        self._seconds += time.perf_counter() - t0
        return samples

    # ------------------------------------------------------------ weights
    def init_weight_sync_group(
        self, master_address: str, master_port: int, rank_offset: int, world_size: int
    ) -> None:
        self.llm.collective_rpc(
            "kiln_init_sync_group", args=(master_address, master_port, rank_offset, world_size)
        )

    def recv_weight_nccl(self, name: str, dtype: torch.dtype, shape: torch.Size) -> None:
        self.llm.collective_rpc("kiln_recv_weight", args=(name, str(dtype), list(shape)))

    def update_weights(self, named_tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        chunk: list[tuple[str, torch.Tensor]] = []
        nbytes = 0
        for name, t in named_tensors:
            chunk.append((name, t.detach().cpu()))
            nbytes += t.numel() * t.element_size()
            if nbytes > (512 << 20):
                self.llm.collective_rpc("kiln_load_cpu_weights", args=(chunk,))
                chunk, nbytes = [], 0
        if chunk:
            self.llm.collective_rpc("kiln_load_cpu_weights", args=(chunk,))
        self.policy_version = version
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self) -> None:
        self.llm.sleep(level=1)

    def wake(self) -> None:
        self.llm.wake_up()

    def stats(self) -> dict[str, float]:
        return {
            "rollout/tokens": float(self._tokens),
            "rollout/tok_per_s": self._tokens / max(self._seconds, 1e-9),
        }
