"""Continuous-batching scheduler with a paged-KV budget, chunked prefill and preemption.

This is the control-plane logic of an LLM engine's scheduler, separated from
any model code so it can be unit-tested and used as a *simulator* for
capacity planning ("how many replicas do I need for 15K RPS at p99 < 500 ms
with this prompt/response distribution?").

Model of one scheduler tick (``step``):

1. **Admission.** Waiting requests are admitted FCFS while (a) the running
   batch has fewer than ``max_num_seqs`` sequences, (b) the prefill token
   budget ``max_num_batched_tokens`` is not exceeded — long prompts are
   *chunked* so one huge prompt cannot stall decode for everyone — and (c)
   enough KV blocks are free for the prompt *and* a conservative reservation
   for the first decode steps.
2. **Decode.** Every running request with finished prefill gets one more
   token, allocating a new block whenever its last block is full.
3. **Preemption.** If decode needs a block and none is free, the *most
   recently admitted* request is evicted (recompute-on-resume, like vLLM)
   and sent back to the head of the waiting queue. Evicting the youngest
   request wastes the least prefill work.

Time is advanced using a simple cost model (``prefill_tok_us``,
``decode_step_us`` + per-seq term) so :func:`simulate` returns latency
percentiles; the scheduling decisions themselves are independent of it.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    num_blocks: int = 8192
    block_size: int = 16
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    chunked_prefill: bool = True
    decode_reserve_blocks: int = 1
    # cost model (per tick) -- defaults loosely calibrated to an 8B model on one H100
    prefill_tok_us: float = 25.0
    decode_step_us: float = 6000.0
    decode_seq_us: float = 40.0


class KVBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self.peak_used = 0

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self._free)

    def can_allocate(self, n: int) -> bool:
        return len(self._free) >= n

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise MemoryError(f"need {n} blocks, {len(self._free)} free")
        out = [self._free.pop() for _ in range(n)]
        self.peak_used = max(self.peak_used, self.used_blocks)
        return out

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)

    def blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_size)


@dataclass(order=True)
class Request:
    arrival: float
    rid: int = field(compare=False)
    prompt_len: int = field(compare=False)
    max_new: int = field(compare=False)
    # runtime state
    prefilled: int = field(default=0, compare=False)
    generated: int = field(default=0, compare=False)
    blocks: list[int] = field(default_factory=list, compare=False)
    admitted_at: float | None = field(default=None, compare=False)
    first_token_at: float | None = field(default=None, compare=False)
    finished_at: float | None = field(default=None, compare=False)
    preemptions: int = field(default=0, compare=False)
    admit_seq: int = field(default=0, compare=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.generated

    @property
    def prefill_done(self) -> bool:
        return self.prefilled >= self.prompt_len

    @property
    def done(self) -> bool:
        return self.generated >= self.max_new

    @property
    def ttft(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.arrival

    @property
    def e2e(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.arrival


@dataclass
class TickStats:
    prefill_tokens: int = 0
    decode_seqs: int = 0
    admitted: int = 0
    preempted: int = 0
    finished: int = 0
    seconds: float = 0.0


class Scheduler:
    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg
        self.kv = KVBlockAllocator(cfg.num_blocks, cfg.block_size)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.now = 0.0
        self._admit_counter = 0
        self.total_preemptions = 0

    # ------------------------------------------------------------ public
    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def step(self) -> TickStats:
        st = TickStats()
        token_budget = self.cfg.max_num_batched_tokens

        # 1. continue chunked prefill of already-running requests first (they hold KV)
        for r in self.running:
            if not r.prefill_done and token_budget > 0:
                chunk = (
                    min(r.prompt_len - r.prefilled, token_budget)
                    if self.cfg.chunked_prefill
                    else r.prompt_len - r.prefilled
                )
                if chunk > token_budget:
                    continue
                r.prefilled += chunk
                token_budget -= chunk
                st.prefill_tokens += chunk

        # 2. admit new requests
        while self.waiting and len(self.running) < self.cfg.max_num_seqs and token_budget > 0:
            r = self.waiting[0]
            need = self.kv.blocks_for(r.prompt_len) + self.cfg.decode_reserve_blocks
            if not self.kv.can_allocate(need):
                break
            chunk = min(r.prompt_len, token_budget) if self.cfg.chunked_prefill else r.prompt_len
            if chunk > token_budget:
                break
            self.waiting.popleft()
            r.blocks = self.kv.allocate(self.kv.blocks_for(r.prompt_len))
            r.prefilled = chunk
            r.admitted_at = self.now
            self._admit_counter += 1
            r.admit_seq = self._admit_counter
            self.running.append(r)
            token_budget -= chunk
            st.prefill_tokens += chunk
            st.admitted += 1

        # 3. decode one token for every prefilled running request, preempting on OOM
        decoders = [r for r in self.running if r.prefill_done]
        for r in list(decoders):
            if r not in self.running:
                continue
            if r.total_tokens % self.cfg.block_size == 0:  # last block full -> need another
                while not self.kv.can_allocate(1):
                    victim = self._preempt_youngest(exclude=r)
                    st.preempted += 1
                    if victim is None:
                        break
                if not self.kv.can_allocate(1):
                    continue  # could not make room; this request stalls one tick
                r.blocks += self.kv.allocate(1)
            r.generated += 1
            st.decode_seqs += 1

        # 4. advance clock with cost model, then stamp completion times
        st.seconds = (
            st.prefill_tokens * self.cfg.prefill_tok_us
            + (self.cfg.decode_step_us + st.decode_seqs * self.cfg.decode_seq_us) * (st.decode_seqs > 0)
        ) / 1e6
        self.now += st.seconds
        for r in list(self.running):
            if r.generated >= 1 and r.first_token_at is None:
                r.first_token_at = self.now
            if r.done:
                r.finished_at = self.now
                self.kv.free(r.blocks)
                r.blocks = []
                self.running.remove(r)
                self.finished.append(r)
                st.finished += 1
        return st

    def _preempt_youngest(self, exclude: Request) -> Request | None:
        candidates = [r for r in self.running if r is not exclude]
        if not candidates:
            return None
        victim = max(candidates, key=lambda r: r.admit_seq)
        self.kv.free(victim.blocks)
        victim.blocks = []
        victim.prefilled = 0
        victim.generated = 0
        victim.first_token_at = None
        victim.preemptions += 1
        self.total_preemptions += 1
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        return victim

    @property
    def idle(self) -> bool:
        return not self.waiting and not self.running

    def kv_utilization(self) -> float:
        return self.kv.used_blocks / self.kv.num_blocks


# ------------------------------------------------------------------ simulator


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(int(round((p / 100) * (len(s) - 1))), len(s) - 1)
    return s[k]


@dataclass
class SimResult:
    requests: int
    duration_s: float
    throughput_rps: float
    output_tok_per_s: float
    ttft_p50: float
    ttft_p99: float
    e2e_p50: float
    e2e_p99: float
    preemptions: int
    peak_kv_util: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def simulate(
    cfg: SchedulerConfig,
    num_requests: int = 2000,
    arrival_rps: float = 50.0,
    prompt_len: tuple[int, int] = (128, 1024),
    max_new: tuple[int, int] = (32, 512),
    seed: int = 0,
    max_ticks: int = 10_000_000,
) -> SimResult:
    """Poisson arrivals, uniform prompt/response lengths; returns latency percentiles."""
    rng = random.Random(seed)
    sched = Scheduler(cfg)
    t = 0.0
    pending: list[Request] = []
    for i in range(num_requests):
        t += rng.expovariate(arrival_rps)
        pending.append(Request(t, i, rng.randint(*prompt_len), rng.randint(*max_new)))
    heapq.heapify(pending)
    ticks = 0
    while (pending or not sched.idle) and ticks < max_ticks:
        while pending and pending[0].arrival <= sched.now:
            sched.add(heapq.heappop(pending))
        if sched.idle and pending:
            sched.now = max(sched.now, pending[0].arrival)
            continue
        sched.step()
        ticks += 1
    fin = sched.finished
    ttft = [r.ttft for r in fin if r.ttft is not None]
    e2e = [r.e2e for r in fin if r.e2e is not None]
    dur = max(sched.now, 1e-9)
    return SimResult(
        requests=len(fin),
        duration_s=dur,
        throughput_rps=len(fin) / dur,
        output_tok_per_s=sum(r.generated for r in fin) / dur,
        ttft_p50=_pct(ttft, 50),
        ttft_p99=_pct(ttft, 99),
        e2e_p50=_pct(e2e, 50),
        e2e_p99=_pct(e2e, 99),
        preemptions=sched.total_preemptions,
        peak_kv_util=sched.kv.peak_used / cfg.num_blocks,
    )
