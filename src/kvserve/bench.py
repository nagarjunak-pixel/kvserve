"""Reproducible serving experiments: each isolates one mechanism against a baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from kvserve.cost import H100, LLAMA3_8B, GPUSpec, ModelSpec, kv_capacity_blocks
from kvserve.engine import EngineConfig, simulate
from kvserve.hashing import BlockHasher
from kvserve.workload import multi_turn_workload, poisson_workload, shared_prefix_workload

KEYS = (
    "throughput_tok_s",
    "goodput_rps",
    "slo_attainment",
    "ttft_p50_s",
    "ttft_p99_s",
    "tpot_p50_s",
    "tpot_p99_s",
    "preemptions",
    "prefix_hit_rate",
    "prefill_tokens_computed",
    "peak_kv_utilization",
    "mean_batch_size",
)


def _row(label: str, summary: dict[str, float]) -> dict[str, Any]:
    return {"config": label, **{k: summary.get(k, 0.0) for k in KEYS}}


def batching_experiment(
    rates: tuple[float, ...] = (10, 30, 50), n: int = 400, model: ModelSpec = LLAMA3_8B, gpu: GPUSpec = H100
) -> list[dict[str, Any]]:
    rows = []
    for rate in rates:
        w = poisson_workload(n=n, rate=rate, seed=1)
        cont = simulate(w, EngineConfig(num_blocks=8000, max_num_seqs=256), model, gpu).summary()
        stat = simulate(
            w, EngineConfig(num_blocks=8000, policy="static", max_num_seqs=64), model, gpu
        ).summary()
        rows.append({"rate_rps": rate, **_row("continuous", cont)})
        rows.append({"rate_rps": rate, **_row("static", stat)})
    return rows


def prefix_experiment(
    n: int = 300, model: ModelSpec = LLAMA3_8B, gpu: GPUSpec = H100
) -> list[dict[str, Any]]:
    rows = []
    for name, w in (
        ("shared-system-prompt", shared_prefix_workload(n=n, rate=12, seed=0)),
        ("multi-turn-agent", multi_turn_workload(conversations=n // 5, turns=5, rate=2.0, seed=0)),
    ):
        on = simulate(w, EngineConfig(num_blocks=8000), model, gpu).summary()
        off = simulate(w, EngineConfig(num_blocks=8000, prefix_caching=False), model, gpu).summary()
        rows.append({"workload": name, **_row("prefix-cache on", on)})
        rows.append({"workload": name, **_row("prefix-cache off", off)})
    return rows


def chunking_experiment(
    rate: float = 8.0, n: int = 300, model: ModelSpec = LLAMA3_8B, gpu: GPUSpec = H100
) -> list[dict[str, Any]]:
    w = poisson_workload(n=n, rate=rate, prompt_mean=3000, output_mean=200, max_prompt=8192, seed=2)
    rows = []
    for label, cfg in (
        ("chunked, budget 512", EngineConfig(num_blocks=12000, max_batched_tokens=512)),
        ("chunked, budget 2048", EngineConfig(num_blocks=12000, max_batched_tokens=2048)),
        (
            "no chunking (prefill-priority), budget 8192",
            EngineConfig(num_blocks=12000, chunked_prefill=False, max_batched_tokens=8192),
        ),
    ):
        rows.append(_row(label, simulate(w, cfg, model, gpu).summary()))
    return rows


def memory_experiment(
    n: int = 300, model: ModelSpec = LLAMA3_8B, gpu: GPUSpec = H100
) -> list[dict[str, Any]]:
    w = poisson_workload(n=n, rate=20, seed=3)
    rows = []
    for blocks in (400, 800, 1600, 8000):
        res = simulate(w, EngineConfig(num_blocks=blocks), model, gpu, check_invariants=True)
        rows.append(
            {
                "num_blocks": blocks,
                "kv_gib": blocks * 16 * model.kv_bytes_per_token / 2**30,
                **_row(f"{blocks} blocks", res.summary()),
            }
        )
    return rows


def kernel_experiment(tokens: int = 200_000) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 32_000, size=tokens, dtype=np.int32)
    native, numpy_h = BlockHasher(16), BlockHasher(16, use_native=False)
    t0 = time.perf_counter()
    a = native.hashes(seq)
    t1 = time.perf_counter()
    b = numpy_h.hashes(seq)
    t2 = time.perf_counter()
    return {
        "tokens": tokens,
        "blocks": int(a.shape[0]),
        "native_ms": (t1 - t0) * 1e3,
        "numpy_ms": (t2 - t1) * 1e3,
        "speedup": (t2 - t1) / (t1 - t0),
        "bit_exact": bool(np.array_equal(a, b)),
        "native_backend": native.backend,
    }


def run_all(quick: bool = False) -> dict[str, Any]:
    n = 120 if quick else 400
    return {
        "setup": {
            "model": LLAMA3_8B.name,
            "gpu": H100.name,
            "kv_bytes_per_token": LLAMA3_8B.kv_bytes_per_token,
            "full_gpu_kv_blocks": kv_capacity_blocks(LLAMA3_8B, H100, 16),
            "slo": {"ttft_s": 2.0, "tpot_s": 0.1},
        },
        "batching": batching_experiment(n=n),
        "prefix_caching": prefix_experiment(n=max(100, n * 3 // 4)),
        "chunked_prefill": chunking_experiment(n=max(100, n * 3 // 4)),
        "memory_pressure": memory_experiment(n=max(100, n * 3 // 4)),
        "kernel": kernel_experiment(50_000 if quick else 200_000),
    }
