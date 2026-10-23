"""Ray-backed deployment of the async loop.

Resource layout
---------------
Two placement strategies, chosen by ``RayConfig.mode``:

``disaggregated`` (default for >1 node)
    Trainer ranks and rollout engines live on *different* GPUs. Rollout
    engines run continuously; weight sync is an NCCL broadcast from trainer
    rank 0 into every engine TP worker. Best throughput, costs extra GPUs.

``colocated``
    Trainer and engine share GPUs. Before each optimizer step the engine
    ``sleep()``s (frees KV cache + weights), the trainer steps, the engine
    ``wake()``s and receives weights over CUDA IPC. Cheapest, but serialises
    rollout and training so it is only "async" across the prompt queue.

Actors
------
* ``BufferActor``   — a single :class:`ExperienceBuffer` behind a Ray handle.
* ``RolloutActor``  — owns one engine (vLLM/SGLang) and runs ``RolloutWorker``.
* ``TrainerActor``  — one per trainer rank; rank 0 also drives the controller.

``launch(cfg)`` builds placement groups, starts everything, runs the controller
on rank 0 and returns its :class:`RunResult`. Requires ``pip install
'kiln-rl[ray]'``; nothing in this module is imported by the local path.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Any

from kiln.config.schema import KilnConfig

log = logging.getLogger("kiln.ray")


@dataclass
class RayConfig:
    mode: str = "disaggregated"  # or "colocated"
    trainer_gpus: int = 8
    gpus_per_engine: int = 1
    num_engines: int = 4
    sync_transport: str = "nccl"
    address: str | None = None


def _ray():
    try:
        import ray
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'kiln-rl[ray]' to use the Ray runtime") from e
    return ray


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def build_actors(cfg: KilnConfig, rcfg: RayConfig):
    """Create placement groups + actors. Returns ``(buffer, rollout_actors, trainer_actors)``."""
    ray = _ray()
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    if not ray.is_initialized():
        ray.init(address=rcfg.address, ignore_reinit_error=True)

    bundles = [{"GPU": 1, "CPU": 4} for _ in range(rcfg.trainer_gpus)]
    strategy = "PACK"
    if rcfg.mode == "disaggregated":
        bundles += [{"GPU": rcfg.gpus_per_engine, "CPU": 8} for _ in range(rcfg.num_engines)]
        strategy = "SPREAD" if rcfg.trainer_gpus > 8 else "PACK"
    pg = placement_group(bundles, strategy=strategy, name="kiln")
    ray.get(pg.ready())

    buffer = _actor("BufferActor").remote(cfg.async_.queue_depth, cfg.async_.max_staleness)

    master_addr = ray.util.get_node_ip_address()
    master_port = _free_port()
    trainers = []
    for rank in range(rcfg.trainer_gpus):
        trainers.append(
            _actor("TrainerActor")
            .options(
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=rank),
            )
            .remote(cfg, rank, rcfg.trainer_gpus, master_addr, master_port)
        )

    rollouts = []
    for i in range(rcfg.num_engines):
        bundle = rcfg.trainer_gpus + i if rcfg.mode == "disaggregated" else i % rcfg.trainer_gpus
        gpus = rcfg.gpus_per_engine if rcfg.mode == "disaggregated" else 0.01
        rollouts.append(
            _actor("RolloutActor")
            .options(
                num_gpus=gpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_bundle_index=bundle),
            )
            .remote(cfg, i, rcfg.num_engines, buffer)
        )
    return buffer, rollouts, trainers


def launch(cfg: KilnConfig, rcfg: RayConfig | None = None) -> dict[str, Any]:
    ray = _ray()
    rcfg = rcfg or RayConfig(num_engines=cfg.rollout.num_engines)
    buffer, rollouts, trainers = build_actors(cfg, rcfg)
    ray.get([t.setup.remote() for t in trainers])
    ray.get([r.setup.remote() for r in rollouts])
    for r in rollouts:
        r.start.remote()
    # rank 0 drives the loop; other ranks follow via torch.distributed collectives inside train_step
    results = ray.get([t.run.remote(rollouts, buffer) for t in trainers])
    ray.get([r.stop.remote() for r in rollouts])
    return results[0]


# --------------------------------------------------------------------- actors
# Decorators are applied lazily so this module imports without Ray installed.


class _BufferActor:
    def __init__(self, capacity: int, max_staleness: int):
        from kiln.orchestration.experience_buffer import ExperienceBuffer

        self.buf = ExperienceBuffer(capacity, max_staleness)

    def put(self, batch, version, timeout=None):
        return self.buf.put(batch, version, timeout)

    def get(self, current_version, timeout=None):
        return self.buf.get(current_version, timeout)

    def resize(self, capacity):
        self.buf.resize(capacity)

    def metrics(self):
        return self.buf.metrics()

    def close(self):
        self.buf.close()

    @property
    def capacity(self):
        return self.buf.capacity


class _RemoteBuffer:
    """Thin sync proxy so the controller/worker code sees the local buffer API."""

    def __init__(self, actor):
        self.actor = actor
        self._ray = _ray()

    def put(self, batch, version, timeout=None):
        return self._ray.get(self.actor.put.remote(batch, version, timeout))

    def get(self, current_version, timeout=None):
        return self._ray.get(self.actor.get.remote(current_version, timeout))

    def resize(self, capacity):
        self._ray.get(self.actor.resize.remote(capacity))

    def metrics(self):
        return self._ray.get(self.actor.metrics.remote())

    def close(self):
        self._ray.get(self.actor.close.remote())

    @property
    def capacity(self):
        return self.metrics()["buffer/capacity"]


class _RolloutActor:
    def __init__(self, cfg: KilnConfig, index: int, total: int, buffer):
        self.cfg, self.index, self.total = cfg, index, total
        self.buffer = _RemoteBuffer(buffer)
        self.worker = None

    def setup(self):
        from kiln.data.prompts import HFTokenizerAdapter, PromptDataset
        from kiln.orchestration.rollout_worker import RolloutWorker
        from kiln.reward import build_reward_fn
        from kiln.rollout import build_engine

        tok = HFTokenizerAdapter(self.cfg.model.name_or_path, chat_template=self.cfg.data.chat_template)
        ds = PromptDataset(self.cfg.data, tok)
        engine = build_engine(
            self.cfg.rollout, model_path=self.cfg.model.name_or_path, eos_id=tok.eos_id, seed=self.index
        )
        ppb = max(self.cfg.train.rollout_batch_size // self.cfg.rollout.sampling.n, 1)
        prompts = ds.batches(
            ppb, seed=self.cfg.train.seed + 1000 * self.index, rank=self.index, world_size=self.total
        )
        self.worker = RolloutWorker(
            self.index, self.cfg, engine, prompts, build_reward_fn(self.cfg.reward), self.buffer, tok, ppb
        )
        self.engine = engine

    def start(self):
        self.worker.start()

    def stop(self):
        self.worker.stop()
        self.engine.shutdown()

    def metrics(self):
        return self.worker.metrics()

    def update_weights(self, named_tensors, version):
        self.engine.update_weights(named_tensors, version)

    def pause(self):
        self.engine.pause()

    def resume(self):
        self.engine.resume()

    def sleep(self):
        self.engine.sleep()

    def wake(self):
        self.engine.wake()

    def error(self):
        return self.worker.error


class _RemoteEngine:
    """Makes a RolloutActor look like a RolloutEngine to :class:`WeightSyncer`."""

    def __init__(self, actor):
        self.actor = actor
        self.policy_version = 0
        self._ray = _ray()

    def update_weights(self, named_tensors, version):
        self._ray.get(self.actor.update_weights.remote(list(named_tensors), version))
        self.policy_version = version

    def pause(self):
        self._ray.get(self.actor.pause.remote())

    def resume(self):
        self._ray.get(self.actor.resume.remote())

    def stats(self):
        return {}

    def shutdown(self):
        pass


class _RemoteWorker:
    def __init__(self, actor, index: int):
        self.actor, self.id = actor, index
        self.engine = _RemoteEngine(actor)
        self._ray = _ray()

    def start(self):
        pass  # started by launch()

    def stop(self, join=True):
        pass

    def metrics(self):
        return self._ray.get(self.actor.metrics.remote())

    @property
    def error(self):
        return self._ray.get(self.actor.error.remote())


class _TrainerActor:
    def __init__(self, cfg: KilnConfig, rank: int, world: int, master_addr: str, master_port: int):
        self.cfg, self.rank, self.world = cfg, rank, world
        os.environ.update(
            {
                "RANK": str(rank),
                "WORLD_SIZE": str(world),
                "LOCAL_RANK": "0",
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
            }
        )

    def setup(self):
        from kiln.data.prompts import HFTokenizerAdapter
        from kiln.distributed import init_parallel
        from kiln.trainer.actor import PolicyTrainer

        self.pstate = init_parallel(self.cfg.parallel)
        self.tok = HFTokenizerAdapter(self.cfg.model.name_or_path)
        self.trainer = PolicyTrainer.build(self.cfg, self.pstate, pad_id=self.tok.pad_id)

    def run(self, rollout_actors, buffer_actor):
        from kiln.checkpoint import AsyncCheckpointer
        from kiln.orchestration.controller import AsyncController
        from kiln.rollout import WeightSyncer
        from kiln.utils.logging import build_logger

        workers = [_RemoteWorker(a, i) for i, a in enumerate(rollout_actors)]
        syncer = WeightSyncer([w.engine for w in workers], transport="cpu")
        ckpt = AsyncCheckpointer(self.cfg.checkpoint)
        logger = build_logger(self.cfg.logging, is_main=self.pstate.is_main)
        ctl = AsyncController(
            self.cfg, self.trainer, workers, _RemoteBuffer(buffer_actor), syncer, ckpt, logger, self.pstate
        )
        res = ctl.run()
        return {"steps": res.steps, "final_metrics": res.final_metrics}


def _actor(name: str):
    return __getattr__(name)


def __getattr__(name: str):
    """Lazily wrap actor classes with ``ray.remote`` on first access."""
    mapping = {"BufferActor": _BufferActor, "RolloutActor": _RolloutActor, "TrainerActor": _TrainerActor}
    if name in mapping:
        ray = _ray()
        wrapped = ray.remote(mapping[name])
        globals()[name] = wrapped
        return wrapped
    raise AttributeError(name)
