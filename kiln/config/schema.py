"""Typed configuration schema for Kiln.

Everything a run needs is expressed in one YAML file validated by these
pydantic models. The schema is intentionally strict (``extra="forbid"``) so a
typo in a researcher-facing YAML fails at load time instead of silently
falling back to a default three hours into a run.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = {"extra": "forbid", "validate_assignment": True}


class ModelConfig(_Strict):
    name_or_path: str = "tiny"
    """HF model id / local path, or ``tiny`` for the built-in test transformer."""
    dtype: Literal["fp32", "bf16", "fp16"] = "bf16"
    attn_impl: Literal["sdpa", "flash", "eager"] = "sdpa"
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    tiny_vocab: int = 512
    tiny_layers: int = 2
    tiny_hidden: int = 64
    tiny_heads: int = 4
    tiny_max_seq: int = 256


class ParallelConfig(_Strict):
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    fsdp_reshard_after_forward: bool = True
    fsdp_cpu_offload: bool = False

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp

    @model_validator(mode="after")
    def _only_implemented(self) -> ParallelConfig:
        # The device mesh can express tp/pp/cp, but no model code shards along them yet
        # (see docs/STATUS.md). Accepting them would silently duplicate work on those ranks.
        unsupported = {k: getattr(self, k) for k in ("tp", "pp", "cp") if getattr(self, k) != 1}
        if unsupported:
            raise ValueError(
                f"{unsupported}: tensor/pipeline/context parallelism is not implemented in the trainer yet; "
                "use data-parallel FSDP2 (parallel.dp) — see docs/STATUS.md"
            )
        return self


class OptimConfig(_Strict):
    lr: float = 1e-6
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 10
    schedule: Literal["constant", "cosine", "linear"] = "constant"
    grad_clip: float = 1.0


class TrainConfig(_Strict):
    total_steps: int = 100
    micro_batch_size: int = 4
    """Sequences per forward/backward on one rank."""
    rollout_batch_size: int = 64
    """Prompts consumed per optimizer step (across all ranks)."""
    ppo_epochs: int = 1
    """How many passes over a rollout batch (1 = strictly on-policy GRPO)."""
    pack_sequences: bool = True
    max_seq_len: int = 4096
    seed: int = 1234
    log_interval: int = 1
    optim: OptimConfig = Field(default_factory=OptimConfig)


class SamplingConfig(_Strict):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 1024
    n: int = 8
    """Responses per prompt (the GRPO group size)."""
    stop: list[str] = Field(default_factory=list)


class RolloutConfig(_Strict):
    engine: Literal["vllm", "sglang", "mock"] = "mock"
    num_engines: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    enable_prefix_caching: bool = True
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    sglang_urls: list[str] = Field(default_factory=list)
    partial_rollout: bool = True
    """Pause in-flight generation, swap weights, resume (instead of draining)."""
    mock_latency_ms_per_token: float = 0.0
    """Mock engine only: add simulated decode time so CPU runs can model a GPU engine's
    wall-clock behaviour. Each batch takes as long as its *longest* simulated response,
    which is what makes synchronous RL wait on the length tail. 0 disables."""
    mock_median_tokens: int = 600
    mock_length_sigma: float = 1.0
    """Simulated response lengths are lognormal(median, sigma); sigma=1 gives p99/p50 ~ 10x,
    in line with reasoning-model traces."""


class AsyncConfig(_Strict):
    enabled: bool = True
    queue_depth: int = 2
    """Max number of rollout *batches* buffered ahead of the trainer."""
    max_staleness: int = 2
    """Drop samples whose policy version lags the trainer by more than this."""
    is_correction: Literal["none", "token", "sequence"] = "token"
    is_clip: float = 2.0
    """Cap on importance ratios used for staleness correction."""
    adaptive_depth: bool = True
    """Let StalenessController grow/shrink queue_depth from measured throughput."""


class KLConfig(_Strict):
    coef: float = 0.001
    estimator: Literal["k1", "k2", "k3"] = "k3"
    in_reward: bool = False
    """PPO-style: subtract KL from reward instead of adding a loss term."""
    target: float | None = None
    """If set, adapt ``coef`` to track this KL (Ziegler et al. controller)."""
    horizon: int = 10_000


class AlgorithmConfig(_Strict):
    name: Literal["grpo", "dr_grpo", "ppo", "rloo", "reinforce_pp", "dpo"] = "grpo"
    clip_eps: float = 0.2
    clip_eps_high: float | None = None
    """Asymmetric clipping (DAPO-style). ``None`` = symmetric."""
    dual_clip: float | None = 3.0
    loss_agg: Literal["token", "sequence", "seq_mean_token_sum"] = "token"
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
    value_clip: float = 0.2
    value_coef: float = 0.5
    dpo_beta: float = 0.1
    dpo_variant: Literal["sigmoid", "ipo", "hinge"] = "sigmoid"
    kl: KLConfig = Field(default_factory=KLConfig)

    @property
    def needs_critic(self) -> bool:
        return self.name == "ppo"

    @property
    def needs_reference(self) -> bool:
        return self.kl.coef > 0 or self.name == "dpo"


class RewardFnConfig(_Strict):
    name: str
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class RewardConfig(_Strict):
    mode: Literal["rules", "server", "model"] = "rules"
    functions: list[RewardFnConfig] = Field(default_factory=lambda: [RewardFnConfig(name="math_boxed")])
    server_url: str | None = None
    server_timeout_s: float = 30.0
    model_name_or_path: str | None = None
    batch_size: int = 64


class DataConfig(_Strict):
    train_path: str = "data/train.jsonl"
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    chat_template: bool = True
    max_prompt_len: int = 1024
    shuffle: bool = True
    num_workers: int = 2


class CheckpointConfig(_Strict):
    dir: str = "checkpoints"
    interval: int = 50
    async_save: bool = True
    keep_last: int = 3
    resume: bool = True
    save_optimizer: bool = True


class LoggingConfig(_Strict):
    backend: Literal["stdout", "wandb", "tensorboard"] = "stdout"
    project: str = "kiln"
    run_name: str | None = None
    log_dir: str = "runs"


class KilnConfig(_Strict):
    model: ModelConfig = Field(default_factory=ModelConfig)
    reference: ModelConfig | None = None
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    async_: AsyncConfig = Field(default_factory=AsyncConfig, alias="async")
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"extra": "forbid", "populate_by_name": True}

    @model_validator(mode="after")
    def _check(self) -> KilnConfig:
        if self.algorithm.name == "dpo" and self.async_.enabled:
            raise ValueError("DPO is offline; set async.enabled=false")
        if self.train.rollout_batch_size % self.train.micro_batch_size != 0:
            raise ValueError("rollout_batch_size must be divisible by micro_batch_size")
        if self.rollout.engine == "sglang" and not self.rollout.sglang_urls:
            raise ValueError("rollout.engine=sglang requires rollout.sglang_urls")
        return self

    @field_validator("reference", mode="before")
    @classmethod
    def _ref_default(cls, v: Any) -> Any:
        return v

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> KilnConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        raw = _expand_env(raw)
        for ov in overrides or []:
            _apply_override(raw, ov)
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(by_alias=True, mode="json"), sort_keys=False)


_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(obj: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in string leaves."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):

        def repl(m: re.Match[str]) -> str:
            val = os.environ.get(m.group(1))
            if val is None:
                if m.group(2) is not None:
                    return m.group(2)
                raise KeyError(f"environment variable {m.group(1)} is not set")
            return val

        return _ENV_RE.sub(repl, obj)
    return obj


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply ``a.b.c=value`` style CLI overrides onto the raw dict."""
    if "=" not in override:
        raise ValueError(f"override must look like key.path=value, got {override!r}")
    key, value = override.split("=", 1)
    parts = key.split(".")
    node = raw
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into non-mapping at {p!r}")
    node[parts[-1]] = yaml.safe_load(value)
