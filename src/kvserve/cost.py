"""Roofline step-time model for one decoder-only transformer forward pass on one accelerator.

Each engine iteration processes a mixed batch: some prefill tokens (chunks of prompts) plus one
token per decoding sequence. Time per iteration is

    max(FLOPs / (peak_flops * mfu), bytes / (hbm_bw * mbu)) + fixed overhead

where the FLOPs are the dense matmuls (2 * params per token) plus attention score/value products
over each token's context, and the bytes are one read of the weights plus a read of every
decoding sequence's KV cache. That captures the core serving trade-off: decode is memory-bound,
so batching more decode sequences is almost free until the KV reads dominate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    params: float  # total parameters
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    dtype_bytes: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V for every layer: 2 * layers * kv_heads * head_dim * dtype bytes (GQA-aware)."""
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes

    @property
    def weight_bytes(self) -> float:
        return self.params * self.dtype_bytes


@dataclass(frozen=True, slots=True)
class GPUSpec:
    name: str
    peak_flops: float  # dense bf16 FLOP/s
    hbm_bandwidth: float  # bytes/s
    hbm_bytes: float
    mfu: float = 0.6  # achievable fraction of peak compute
    mbu: float = 0.8  # achievable fraction of peak bandwidth
    overhead_s: float = 0.002  # per-iteration launch / scheduling overhead


LLAMA3_8B = ModelSpec("llama-3-8b", 8.03e9, 32, 4096, 32, 8)
LLAMA3_70B = ModelSpec("llama-3-70b", 70.6e9, 80, 8192, 64, 8)
H100 = GPUSpec("h100-sxm", 989e12, 3.35e12, 80e9)
A100 = GPUSpec("a100-80g", 312e12, 2.0e12, 80e9)


@dataclass(frozen=True, slots=True)
class StepCost:
    seconds: float
    flops: float
    bytes: float
    compute_bound: bool


def step_cost(
    model: ModelSpec,
    gpu: GPUSpec,
    prefill_chunks: Sequence[tuple[int, int]],
    decode_contexts: Sequence[int],
) -> StepCost:
    """Cost of one iteration.

    ``prefill_chunks``: (chunk_tokens, tokens_already_cached_before_chunk) per prefilling request.
    ``decode_contexts``: current context length of each decoding sequence (one new token each).
    """
    d = model.hidden
    tokens = sum(c for c, _ in prefill_chunks) + len(decode_contexts)
    if tokens == 0:
        return StepCost(0.0, 0.0, 0.0, False)
    dense = 2.0 * model.params * tokens
    attn = 0.0
    for chunk, cached in prefill_chunks:
        # each new token attends to cached + preceding new tokens (causal): QK^T and AV
        attn += 4.0 * model.layers * d * (chunk * cached + chunk * (chunk + 1) / 2.0)
    for ctx in decode_contexts:
        attn += 4.0 * model.layers * d * ctx
    flops = dense + attn
    kv_read = model.kv_bytes_per_token * (sum(decode_contexts) + sum(cached for _, cached in prefill_chunks))
    kv_write = model.kv_bytes_per_token * tokens
    nbytes = model.weight_bytes + kv_read + kv_write
    t_compute = flops / (gpu.peak_flops * gpu.mfu)
    t_memory = nbytes / (gpu.hbm_bandwidth * gpu.mbu)
    return StepCost(max(t_compute, t_memory) + gpu.overhead_s, flops, nbytes, t_compute >= t_memory)


def kv_capacity_blocks(model: ModelSpec, gpu: GPUSpec, block_size: int, memory_fraction: float = 0.9) -> int:
    """KV blocks that fit after the weights (vLLM's gpu_memory_utilization style budget)."""
    free = gpu.hbm_bytes * memory_fraction - model.weight_bytes
    if free <= 0:
        raise ValueError(f"{model.name} weights do not fit on {gpu.name}")
    return int(free // (model.kv_bytes_per_token * block_size))
