from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.mock_engine import MockEngine
from kiln.rollout.types import RolloutBatch, Sample, collate_samples, concat_batches
from kiln.rollout.weight_sync import SyncStats, WeightSyncer, iter_sync_tensors, lora_only


def build_engine(
    cfg, model=None, model_path: str | None = None, eos_id: int | None = None, seed: int = 0
) -> RolloutEngine:
    if cfg.engine == "mock":
        assert model is not None, "mock engine needs a model"
        return MockEngine(
            model,
            eos_id=eos_id,
            seed=seed,
            latency_ms_per_token=cfg.mock_latency_ms_per_token,
            median_tokens=cfg.mock_median_tokens,
            length_sigma=cfg.mock_length_sigma,
        )
    if cfg.engine == "vllm":
        from kiln.rollout.vllm_engine import VLLMEngine

        assert model_path is not None
        return VLLMEngine(model_path, cfg, seed=seed)
    if cfg.engine == "sglang":
        from kiln.rollout.sglang_engine import SGLangEngine

        return SGLangEngine(cfg)
    raise ValueError(f"unknown rollout engine {cfg.engine!r}")


__all__ = [
    "GenerationRequest",
    "MockEngine",
    "RolloutBatch",
    "RolloutEngine",
    "Sample",
    "SyncStats",
    "WeightSyncer",
    "build_engine",
    "collate_samples",
    "concat_batches",
    "iter_sync_tensors",
    "lora_only",
]
