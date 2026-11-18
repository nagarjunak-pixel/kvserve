"""Iteration-level LLM serving engine simulator.

Policies:

* ``continuous``: Orca-style iteration-level scheduling (Yu et al., 2022), as in vLLM v1.
  Running sequences are scheduled first (decode-first). Waiting requests are then admitted FCFS
  under a per-step token budget and a sequence cap. With ``chunked_prefill`` long prompts are
  split across steps so decodes are never stalled behind a big prefill (Sarathi-Serve).
  When KV blocks run out, the most recently admitted sequence is preempted and recomputed later.
* ``static``: request-level batching. A batch is admitted only when the previous batch has
  fully drained, the classic pre-Orca baseline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from kvserve.blocks import BlockManager, OutOfBlocksError
from kvserve.cost import GPUSpec, ModelSpec, step_cost
from kvserve.hashing import BlockHasher
from kvserve.workload import Request


@dataclass(frozen=True, slots=True)
class EngineConfig:
    num_blocks: int = 4096
    block_size: int = 16
    max_num_seqs: int = 128
    max_batched_tokens: int = 2048
    chunked_prefill: bool = True
    prefix_caching: bool = True
    policy: str = "continuous"  # or "static"
    watermark_blocks: int = 8  # head-room kept free when admitting, to limit preemption thrash

    def __post_init__(self) -> None:
        if self.policy not in ("continuous", "static"):
            raise ValueError("policy must be 'continuous' or 'static'")
        if min(self.num_blocks, self.block_size, self.max_num_seqs, self.max_batched_tokens) < 1:
            raise ValueError("sizes must be positive")


@dataclass(slots=True)
class _Seq:
    req: Request
    computed: int = 0  # prompt tokens whose KV is materialised
    generated: int = 0
    first_token_time: float | None = None
    finish_time: float | None = None
    preemptions: int = 0
    cached_tokens: int = 0
    tokens: list[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    req_id: int
    arrival: float
    ttft: float
    tpot: float
    e2e: float
    prompt_len: int
    output_len: int
    cached_tokens: int
    preemptions: int


@dataclass(frozen=True, slots=True)
class SimResult:
    requests: tuple[RequestMetrics, ...]
    rejected: tuple[int, ...]
    makespan: float
    steps: int
    preemptions: int
    prefix_hit_rate: float
    prefill_tokens_computed: int
    peak_kv_utilization: float
    mean_batch_size: float

    def summary(self, ttft_slo: float = 2.0, tpot_slo: float = 0.1) -> dict[str, float]:
        if not self.requests:
            return {"completed": 0.0}
        ttft = np.array([r.ttft for r in self.requests])
        tpot = np.array([r.tpot for r in self.requests])
        e2e = np.array([r.e2e for r in self.requests])
        out_tokens = sum(r.output_len for r in self.requests)
        good = sum(1 for r in self.requests if r.ttft <= ttft_slo and r.tpot <= tpot_slo)
        return {
            "completed": float(len(self.requests)),
            "rejected": float(len(self.rejected)),
            "makespan_s": self.makespan,
            "throughput_tok_s": out_tokens / self.makespan if self.makespan > 0 else 0.0,
            "request_throughput_rps": len(self.requests) / self.makespan if self.makespan > 0 else 0.0,
            "goodput_rps": good / self.makespan if self.makespan > 0 else 0.0,
            "slo_attainment": good / len(self.requests),
            "ttft_p50_s": float(np.percentile(ttft, 50)),
            "ttft_p99_s": float(np.percentile(ttft, 99)),
            "tpot_p50_s": float(np.percentile(tpot, 50)),
            "tpot_p99_s": float(np.percentile(tpot, 99)),
            "e2e_p50_s": float(np.percentile(e2e, 50)),
            "e2e_mean_s": float(e2e.mean()),
            "preemptions": float(self.preemptions),
            "prefix_hit_rate": self.prefix_hit_rate,
            "prefill_tokens_computed": float(self.prefill_tokens_computed),
            "peak_kv_utilization": self.peak_kv_utilization,
            "mean_batch_size": self.mean_batch_size,
            "steps": float(self.steps),
        }


class Engine:
    def __init__(
        self, config: EngineConfig, model: ModelSpec, gpu: GPUSpec, hasher: BlockHasher | None = None
    ) -> None:
        self.cfg, self.model, self.gpu = config, model, gpu
        self.bm = BlockManager(
            config.num_blocks,
            config.block_size,
            config.prefix_caching,
            hasher or BlockHasher(config.block_size),
        )
        self.waiting: deque[_Seq] = deque()
        self.running: list[_Seq] = []
        self.done: list[_Seq] = []
        self.rejected: list[int] = []
        self.now = 0.0
        self.steps = 0
        self.preemptions = 0
        self.prefill_computed = 0
        self.peak_util = 0.0
        self.batch_sizes: list[int] = []

    # ----------------------------------------------------------------------------- helpers

    def _preempt(self, seq: _Seq) -> None:
        """Recompute-style preemption: drop KV, restart prefill later (front of the queue)."""
        self.running.remove(seq)
        self.bm.free_seq(seq.req.req_id)
        seq.computed = 0
        seq.preemptions += 1
        self.preemptions += 1
        self.waiting.appendleft(seq)

    def _admit(self, seq: _Seq) -> None:
        # A preempted sequence re-prefills prompt + generated-so-far (recompute semantics).
        seq.tokens = list(seq.req.prompt) + list(seq.req.output[: seq.generated])
        seq.cached_tokens = max(seq.cached_tokens, self.bm.admit(seq.req.req_id, seq.tokens))
        seq.computed = self.bm.seqs[seq.req.req_id].num_tokens
        self.running.append(seq)

    def _chunk(self, seq: _Seq, budget: int, batch_empty: bool) -> int:
        """Tokens of ``seq`` to process this step (0 = skip). Every sequence has
        ``len(tokens) - computed >= 1`` pending tokens; decode is the 1-pending case."""
        remaining = len(seq.tokens) - seq.computed
        if budget <= 0:
            return 0
        if self.cfg.chunked_prefill:
            return min(remaining, budget)
        return remaining if remaining <= budget or batch_empty else 0

    # -------------------------------------------------------------------------------- step

    def _admission(self, budget: int, batch: list[tuple[_Seq, int]]) -> int:
        """Admit waiting requests FCFS while memory, the sequence cap and the budget allow."""
        static = self.cfg.policy == "static"
        while self.waiting and len(self.running) < self.cfg.max_num_seqs and (budget > 0 or static):
            seq = self.waiting[0]
            tokens_all = list(seq.req.prompt) + list(seq.req.output[: seq.generated])
            if not self.bm.can_admit(tokens_all, reserve=self.cfg.watermark_blocks if self.running else 0):
                break
            self.waiting.popleft()
            self._admit(seq)
            chunk = self._chunk(seq, budget, not batch)
            if chunk == 0:
                continue  # admitted; its prefill starts in a later step
            self.bm.append_tokens(seq.req.req_id, seq.tokens[seq.computed : seq.computed + chunk], seq.tokens)
            batch.append((seq, chunk))
            budget -= chunk
        return budget

    def _schedule(self) -> list[tuple[_Seq, int]]:
        budget = self.cfg.max_batched_tokens
        batch: list[tuple[_Seq, int]] = []
        if self.cfg.policy == "continuous" and not self.cfg.chunked_prefill:
            # vLLM-v0-style prefill priority: new prompts run in a prefill-only step and every
            # decode stalls for that step. This is the stall Sarathi-style chunking removes.
            self._admission(budget, batch)
            if batch:
                return batch
        idx = 0
        while idx < len(self.running) and budget > 0:
            seq = self.running[idx]
            chunk = self._chunk(seq, budget, not batch)
            if chunk == 0:
                idx += 1
                continue
            try:
                self.bm.append_tokens(
                    seq.req.req_id, seq.tokens[seq.computed : seq.computed + chunk], seq.tokens
                )
            except OutOfBlocksError:
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    break
                continue
            batch.append((seq, chunk))
            budget -= chunk
            idx += 1
        if (
            self.cfg.chunked_prefill
            and (self.cfg.policy == "continuous" or not self.running)
            or self.cfg.policy == "static"
            and not self.running
        ):
            self._admission(budget, batch)
        return batch

    def step(self) -> None:
        batch = self._schedule()
        if not batch:
            raise RuntimeError("scheduler made no progress")  # pragma: no cover - guarded by admission checks
        prefills = [(chunk, seq.computed) for seq, chunk in batch if not (chunk == 1 and seq.generated)]
        decodes = [len(seq.tokens) for seq, chunk in batch if chunk == 1 and seq.generated]
        cost = step_cost(self.model, self.gpu, prefills, decodes)
        self.now += cost.seconds
        self.steps += 1
        self.batch_sizes.append(len(batch))
        self.peak_util = max(self.peak_util, self.bm.used / self.bm.num_blocks)
        for seq, chunk in batch:
            if not seq.generated or chunk > 1:
                self.prefill_computed += chunk
            seq.computed += chunk
            if seq.computed == len(seq.tokens):
                self._emit(seq)

    def _emit(self, seq: _Seq) -> None:
        """All known tokens now have KV: sample the next output token."""
        token = seq.req.output[seq.generated]
        seq.generated += 1
        if seq.first_token_time is None:
            seq.first_token_time = self.now
        seq.tokens.append(token)
        if seq.generated >= seq.req.max_new_tokens:
            seq.finish_time = self.now
            self.running.remove(seq)
            self.bm.free_seq(seq.req.req_id)
            self.done.append(seq)

    # ---------------------------------------------------------------------------------- run

    def run(self, requests: list[Request], check_invariants: bool = False) -> SimResult:
        pending = deque(sorted(requests, key=lambda r: (r.arrival, r.req_id)))
        while pending or self.waiting or self.running:
            while pending and pending[0].arrival <= self.now:
                req = pending.popleft()
                if self.bm.blocks_needed(req.prompt_len + req.max_new_tokens) > self.bm.num_blocks:
                    self.rejected.append(req.req_id)
                    continue
                self.waiting.append(_Seq(req))
            if not self.waiting and not self.running:
                if pending:
                    self.now = pending[0].arrival
                continue
            self.step()
            if check_invariants:
                self.bm.check_invariants()
        metrics = []
        for s in sorted(self.done, key=lambda s: s.req.req_id):
            assert s.first_token_time is not None and s.finish_time is not None
            n = s.req.max_new_tokens
            metrics.append(
                RequestMetrics(
                    req_id=s.req.req_id,
                    arrival=s.req.arrival,
                    ttft=s.first_token_time - s.req.arrival,
                    tpot=(s.finish_time - s.first_token_time) / (n - 1) if n > 1 else 0.0,
                    e2e=s.finish_time - s.req.arrival,
                    prompt_len=s.req.prompt_len,
                    output_len=n,
                    cached_tokens=s.cached_tokens,
                    preemptions=s.preemptions,
                )
            )
        start = min((r.arrival for r in requests), default=0.0)
        return SimResult(
            requests=tuple(metrics),
            rejected=tuple(self.rejected),
            makespan=self.now - start,
            steps=self.steps,
            preemptions=self.preemptions,
            prefix_hit_rate=self.bm.hit_rate,
            prefill_tokens_computed=self.prefill_computed,
            peak_kv_utilization=self.peak_util,
            mean_batch_size=float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
        )


def simulate(
    requests: list[Request],
    config: EngineConfig,
    model: ModelSpec,
    gpu: GPUSpec,
    check_invariants: bool = False,
) -> SimResult:
    return Engine(config, model, gpu).run(requests, check_invariants)
