"""Prefix-cache-aware request router.

Problem: with N vLLM/SGLang replicas each holding its own prefix cache, naive
round-robin sends requests sharing a long system prompt (or a multi-turn
conversation) to a replica that has never seen the prefix, wasting the cache
and re-doing prefill. Pure sticky hashing fixes cache hit-rate but creates hot
spots.

Policy here: score each replica as

    score = hit_tokens(prefix) * cache_weight - queue_len * load_weight

where ``hit_tokens`` comes from a per-replica radix tree mirror of what we
*sent* there (an approximation of the engine's real cache, decayed by LRU
capacity). Pick the max; ties go to the least-loaded. This is the same idea
as SGLang's cache-aware router, kept small enough to read in one sitting.

The radix tree stores token ids at block granularity so it stays cheap even
for long prompts.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


class _RadixNode:
    __slots__ = ("children", "last_access", "count")

    def __init__(self) -> None:
        self.children: dict[tuple[int, ...], _RadixNode] = {}
        self.last_access = 0.0
        self.count = 0


class BlockRadixTree:
    """Prefix tree over fixed-size token blocks with LRU eviction by block count."""

    def __init__(self, block_size: int = 16, capacity_blocks: int = 4096):
        self.block_size = block_size
        self.capacity = capacity_blocks
        self.root = _RadixNode()
        self._lru: OrderedDict[tuple[tuple[int, ...], ...], None] = OrderedDict()
        self.size_blocks = 0

    def _blocks(self, ids: list[int]) -> list[tuple[int, ...]]:
        n = len(ids) // self.block_size
        return [tuple(ids[i * self.block_size : (i + 1) * self.block_size]) for i in range(n)]

    def match(self, ids: list[int]) -> int:
        """Number of leading tokens already present."""
        node, matched = self.root, 0
        for b in self._blocks(ids):
            nxt = node.children.get(b)
            if nxt is None:
                break
            node = nxt
            matched += self.block_size
        return matched

    def insert(self, ids: list[int]) -> int:
        """Insert; returns number of *new* blocks added."""
        node, path, added = self.root, [], 0
        now = time.monotonic()
        for b in self._blocks(ids):
            path.append(b)
            nxt = node.children.get(b)
            if nxt is None:
                nxt = _RadixNode()
                node.children[b] = nxt
                added += 1
                self.size_blocks += 1
            nxt.last_access = now
            nxt.count += 1
            key = tuple(path)
            self._lru.pop(key, None)
            self._lru[key] = None
            node = nxt
        self._evict()
        return added

    def _evict(self) -> None:
        while self.size_blocks > self.capacity and self._lru:
            path, _ = self._lru.popitem(last=False)
            node = self.root
            for b in path[:-1]:
                node = node.children.get(b)
                if node is None:
                    break
            if node is not None and path[-1] in node.children and not node.children[path[-1]].children:
                del node.children[path[-1]]
                self.size_blocks -= 1


@dataclass
class EngineReplica:
    name: str
    url: str
    capacity_blocks: int = 4096
    block_size: int = 16
    healthy: bool = True
    in_flight: int = 0
    total_requests: int = 0
    cache_hits_tokens: int = 0
    prompt_tokens: int = 0
    tree: BlockRadixTree = field(init=False)

    def __post_init__(self) -> None:
        self.tree = BlockRadixTree(self.block_size, self.capacity_blocks)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits_tokens / max(self.prompt_tokens, 1)


@dataclass
class RouteDecision:
    replica: EngineReplica
    matched_tokens: int
    reason: str


class PrefixRouter:
    def __init__(
        self,
        replicas: list[EngineReplica],
        cache_weight: float = 1.0,
        load_weight: float = 64.0,
        max_imbalance: int = 8,
    ):
        self.replicas = replicas
        self.cache_weight = cache_weight
        self.load_weight = load_weight
        self.max_imbalance = max_imbalance
        self._lock = threading.Lock()

    def route(self, prompt_ids: list[int]) -> RouteDecision:
        with self._lock:
            live = [r for r in self.replicas if r.healthy]
            if not live:
                raise RuntimeError("no healthy replicas")
            min_load = min(r.in_flight for r in live)
            best, best_score, best_match, reason = None, float("-inf"), 0, "least-load"
            for r in live:
                m = r.tree.match(prompt_ids)
                # hard cap on imbalance: a replica far above the least-loaded one is ineligible
                if r.in_flight - min_load > self.max_imbalance:
                    continue
                score = m * self.cache_weight - r.in_flight * self.load_weight
                if score > best_score or (
                    score == best_score and best is not None and r.in_flight < best.in_flight
                ):
                    best, best_score, best_match = r, score, m
            if best is None:
                best = min(live, key=lambda r: r.in_flight)
                best_match = best.tree.match(prompt_ids)
            if best_match > 0:
                reason = "prefix-hit"
            best.in_flight += 1
            best.total_requests += 1
            best.prompt_tokens += len(prompt_ids)
            best.cache_hits_tokens += best_match
            best.tree.insert(prompt_ids)
            return RouteDecision(best, best_match, reason)

    def release(self, replica: EngineReplica) -> None:
        with self._lock:
            replica.in_flight = max(replica.in_flight - 1, 0)

    def mark(self, name: str, healthy: bool) -> None:
        with self._lock:
            for r in self.replicas:
                if r.name == name:
                    r.healthy = healthy

    def stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            return {
                r.name: {
                    "in_flight": r.in_flight,
                    "requests": r.total_requests,
                    "hit_rate": r.hit_rate,
                    "cache_blocks": r.tree.size_blocks,
                    "healthy": float(r.healthy),
                }
                for r in self.replicas
            }
