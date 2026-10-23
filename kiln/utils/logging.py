"""Metric sinks: stdout (rich table), Weights & Biases, TensorBoard, JSONL.

Every backend receives the same flat ``{"group/name": float}`` dict once per
optimizer step. A JSONL copy is always written under ``log_dir/<run>/metrics.jsonl``
so a run can be post-processed without any third-party service, and the
end-to-end tests assert on it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kiln.config.schema import LoggingConfig

log = logging.getLogger("kiln")


def setup_logging(level: str = "INFO") -> None:
    if log.handlers:
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except ImportError:  # pragma: no cover
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(os.environ.get("KILN_LOG_LEVEL", level))
    log.propagate = False


class MetricLogger:
    def __init__(self, cfg: LoggingConfig, run_dir: Path, is_main: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        self.is_main = is_main
        self._jsonl = None
        self._wandb = None
        self._tb = None
        self._t0 = time.time()
        if not is_main:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (run_dir / "metrics.jsonl").open("a")
        if cfg.backend == "wandb":
            import wandb

            self._wandb = wandb.init(project=cfg.project, name=cfg.run_name, dir=str(run_dir), resume="allow")
        elif cfg.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(run_dir / "tb"))

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        if not self.is_main:
            return
        rec = {"step": step, "time": time.time() - self._t0, **{k: _f(v) for k, v in metrics.items()}}
        assert self._jsonl is not None
        self._jsonl.write(json.dumps(rec) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(dict(rec), step=step)
        if self._tb is not None:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k not in ("step",):
                    self._tb.add_scalar(k, v, step)
        log.info(format_metrics(step, rec))

    def close(self) -> None:
        if self._jsonl is not None:
            self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()


_KEY_ORDER = (
    "reward/mean",
    "loss/total",
    "policy/clipfrac",
    "policy/approx_kl",
    "kl/ref",
    "async/mean_staleness",
    "async/queue_depth",
    "buffer/fill",
    "throughput/tokens_per_s",
    "perf/mfu",
    "time/step_s",
)


def format_metrics(step: int, rec: Mapping[str, Any]) -> str:
    parts = [f"step {step:>5}"]
    for k in _KEY_ORDER:
        if k in rec:
            v = rec[k]
            parts.append(f"{k.split('/')[-1]}={_fmt(v)}")
    return " | ".join(parts)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4g}"
    return str(v)


def _f(v: Any) -> Any:
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return float(v.item())
    except ImportError:  # pragma: no cover
        pass
    return float(v) if isinstance(v, (int, float)) else v


def build_logger(cfg: LoggingConfig, is_main: bool = True) -> MetricLogger:
    name = cfg.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    return MetricLogger(cfg, Path(cfg.log_dir) / name, is_main=is_main)


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
