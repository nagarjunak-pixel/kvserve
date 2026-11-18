"""``kvserve`` command-line interface and HTTP API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from kvserve import __version__
from kvserve.bench import run_all
from kvserve.cost import A100, H100, LLAMA3_8B, kv_capacity_blocks
from kvserve.engine import EngineConfig, simulate
from kvserve.workload import multi_turn_workload, poisson_workload, shared_prefix_workload

app = typer.Typer(
    name="kvserve",
    help="kvserve: paged-KV LLM serving simulator.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
GPUS = {"h100": H100, "a100": A100}


class SimRequest(BaseModel):
    workload: Literal["poisson", "shared-prefix", "multi-turn"] = "poisson"
    n: int = Field(default=200, ge=1, le=5000)
    rate: float = Field(default=10.0, gt=0, le=500)
    gpu: Literal["h100", "a100"] = "h100"
    num_blocks: int | None = Field(default=None, ge=16)
    max_num_seqs: int = Field(default=128, ge=1, le=2048)
    max_batched_tokens: int = Field(default=2048, ge=16, le=65536)
    chunked_prefill: bool = True
    prefix_caching: bool = True
    policy: Literal["continuous", "static"] = "continuous"
    seed: int = 0


def run_sim(req: SimRequest) -> dict[str, Any]:
    gpu = GPUS[req.gpu]
    if req.workload == "poisson":
        w = poisson_workload(n=req.n, rate=req.rate, seed=req.seed)
    elif req.workload == "shared-prefix":
        w = shared_prefix_workload(n=req.n, rate=req.rate, seed=req.seed)
    else:
        w = multi_turn_workload(conversations=max(1, req.n // 5), rate=req.rate, seed=req.seed)
    blocks = req.num_blocks or kv_capacity_blocks(LLAMA3_8B, gpu, 16)
    cfg = EngineConfig(
        num_blocks=blocks,
        max_num_seqs=req.max_num_seqs,
        max_batched_tokens=req.max_batched_tokens,
        chunked_prefill=req.chunked_prefill,
        prefix_caching=req.prefix_caching,
        policy=req.policy,
    )
    return {"num_blocks": blocks, **simulate(w, cfg, LLAMA3_8B, gpu).summary()}


def create_app() -> FastAPI:
    api = FastAPI(title="kvserve", version=__version__)

    @api.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.post("/v1/simulate")
    def simulate_endpoint(req: SimRequest) -> dict[str, Any]:
        try:
            return run_sim(req)
        except ValueError as exc:  # pragma: no cover - pydantic bounds keep configs valid
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return api


@app.command("simulate")
def simulate_cmd(
    workload: str = typer.Option("poisson", help="poisson | shared-prefix | multi-turn"),
    n: int = typer.Option(200, min=1),
    rate: float = typer.Option(10.0, min=0.01),
    num_blocks: int | None = typer.Option(None, min=16),
    policy: str = typer.Option("continuous", help="continuous | static"),
    no_chunked_prefill: bool = typer.Option(False, "--no-chunked-prefill"),
    no_prefix_caching: bool = typer.Option(False, "--no-prefix-caching"),
    max_batched_tokens: int = typer.Option(2048, min=16),
) -> None:
    """Simulate one configuration and print its latency / throughput summary as JSON."""
    req = SimRequest.model_validate(
        {
            "workload": workload,
            "n": n,
            "rate": rate,
            "num_blocks": num_blocks,
            "policy": policy,
            "chunked_prefill": not no_chunked_prefill,
            "prefix_caching": not no_prefix_caching,
            "max_batched_tokens": max_batched_tokens,
        }
    )
    typer.echo(json.dumps(run_sim(req), indent=2))


def _table(title: str, rows: list[dict[str, Any]], lead: str) -> Table:
    t = Table(title=title)
    cols = [
        lead,
        "config",
        "tok/s",
        "goodput rps",
        "TTFT p50",
        "TTFT p99",
        "TPOT p99",
        "prefill tok",
        "hit rate",
        "preempt",
    ]
    for c in cols:
        t.add_column(c)
    for r in rows:
        t.add_row(
            str(r.get(lead, "")),
            r["config"],
            f"{r['throughput_tok_s']:.0f}",
            f"{r['goodput_rps']:.2f}",
            f"{r['ttft_p50_s']:.3f}",
            f"{r['ttft_p99_s']:.3f}",
            f"{r['tpot_p99_s']:.3f}",
            f"{r['prefill_tokens_computed']:.0f}",
            f"{r['prefix_hit_rate']:.2f}",
            f"{r['preemptions']:.0f}",
        )
    return t


@app.command("bench")
def bench_cmd(
    quick: bool = typer.Option(False, help="Smaller workloads (CI smoke run)"),
    output: Path | None = typer.Option(None, help="Write the JSON report here"),
) -> None:
    """Run every experiment (batching, prefix caching, chunked prefill, memory pressure, kernel)."""
    r = run_all(quick=quick)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    console.print(_table("Continuous vs static batching", r["batching"], "rate_rps"))
    console.print(_table("Automatic prefix caching", r["prefix_caching"], "workload"))
    console.print(_table("Chunked prefill (long prompts, 8 rps)", r["chunked_prefill"], "config"))
    console.print(_table("KV memory pressure (recompute preemption)", r["memory_pressure"], "num_blocks"))
    k = r["kernel"]
    console.print(
        f"Block-hash kernel ({k['native_backend']}): {k['tokens']} tokens -> {k['blocks']} blocks in "
        f"{k['native_ms']:.2f} ms vs {k['numpy_ms']:.1f} ms NumPy twin ({k['speedup']:.0f}x), bit-exact={k['bit_exact']}"
    )


@app.command("serve")
def serve_cmd(host: str = "127.0.0.1", port: int = 8099) -> None:  # pragma: no cover - blocking
    """Serve the HTTP API."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
