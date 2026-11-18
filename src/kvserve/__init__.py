"""kvserve: paged-KV LLM serving engine simulator.

PagedAttention block management with refcounted automatic prefix caching, iteration-level
(continuous) batching, chunked prefill, recompute preemption and a roofline step-cost model.
"""

__version__ = "1.0.0"

from kvserve.blocks import BlockManager, OutOfBlocksError  # noqa: E402
from kvserve.cost import A100, H100, LLAMA3_8B, LLAMA3_70B, GPUSpec, ModelSpec, kv_capacity_blocks, step_cost  # noqa: E402
from kvserve.engine import Engine, EngineConfig, SimResult, simulate  # noqa: E402
from kvserve.hashing import BlockHasher  # noqa: E402
from kvserve.workload import Request, multi_turn_workload, poisson_workload, shared_prefix_workload  # noqa: E402

__all__ = [
    "A100",
    "H100",
    "LLAMA3_70B",
    "LLAMA3_8B",
    "BlockHasher",
    "BlockManager",
    "Engine",
    "EngineConfig",
    "GPUSpec",
    "ModelSpec",
    "OutOfBlocksError",
    "Request",
    "SimResult",
    "__version__",
    "kv_capacity_blocks",
    "multi_turn_workload",
    "poisson_workload",
    "shared_prefix_workload",
    "simulate",
    "step_cost",
]
