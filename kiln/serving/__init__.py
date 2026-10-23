"""Inference-side infrastructure: routing, scheduling and rollout safety.

* :mod:`kiln.serving.router`     — prefix-cache-aware, load-balanced request router
  over N engine replicas (consistent hashing on the prompt prefix + least-load fallback).
* :mod:`kiln.serving.scheduler`  — continuous-batching scheduler with a KV-cache
  block budget and chunked prefill; also a discrete-event simulator for capacity planning.
* :mod:`kiln.serving.canary`     — SLO guardrails for canary / shadow deployments.
* :mod:`kiln.serving.gateway`    — FastAPI front door that uses the router.
"""

from kiln.serving.canary import SLO, CanaryController, SLOVerdict
from kiln.serving.router import EngineReplica, PrefixRouter
from kiln.serving.scheduler import KVBlockAllocator, Request, Scheduler, SchedulerConfig, simulate

__all__ = [
    "SLO",
    "CanaryController",
    "EngineReplica",
    "KVBlockAllocator",
    "PrefixRouter",
    "Request",
    "SLOVerdict",
    "Scheduler",
    "SchedulerConfig",
    "simulate",
]
