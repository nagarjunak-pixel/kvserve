from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from kvserve import __version__
from kvserve.cli import app as cli_app
from kvserve.cli import create_app
from kvserve.cost import H100, LLAMA3_8B
from kvserve.engine import Engine, EngineConfig, SimResult, simulate
from kvserve.workload import Request, multi_turn_workload, poisson_workload, shared_prefix_workload


def test_workloads_are_deterministic_and_shaped() -> None:
    a, b = poisson_workload(n=50, seed=3), poisson_workload(n=50, seed=3)
    assert a == b and all(r.prompt_len >= 8 and r.max_new_tokens >= 1 for r in a)
    assert all(x.arrival <= y.arrival for x, y in zip(a, a[1:], strict=False))
    sp = shared_prefix_workload(n=20, num_prefixes=2, prefix_len=64)
    assert len({r.prompt[:64] for r in sp}) <= 2
    mt = multi_turn_workload(conversations=3, turns=3)
    assert [r.req_id for r in mt] == list(range(9))
    longest = max(mt, key=lambda r: r.prompt_len)
    assert any(longest.prompt[: r.prompt_len] == r.prompt for r in mt if r is not longest)


def test_engine_completes_under_memory_pressure_with_invariants() -> None:
    w = poisson_workload(n=80, rate=30, seed=1)
    res = simulate(w, EngineConfig(num_blocks=300), LLAMA3_8B, H100, check_invariants=True)
    assert len(res.requests) == 80 and res.preemptions > 0
    assert res.peak_kv_utilization <= 1.0
    for m in res.requests:
        assert m.ttft > 0 and m.e2e >= m.ttft and m.tpot >= 0


def test_generated_tokens_match_requests_even_after_preemption() -> None:
    w = poisson_workload(n=40, rate=40, seed=2)
    eng = Engine(EngineConfig(num_blocks=200, watermark_blocks=0), LLAMA3_8B, H100)
    res = eng.run(w, check_invariants=True)
    assert res.preemptions > 0
    by_id = {r.req_id: r for r in w}
    for seq in eng.done:
        req = by_id[seq.req.req_id]
        assert seq.tokens == list(req.prompt) + list(req.output)
    assert eng.bm.used == 0  # no leaked blocks once everything finished


def test_continuous_batching_beats_static() -> None:
    w = poisson_workload(n=150, rate=30, seed=4)
    cont = simulate(w, EngineConfig(num_blocks=8000, max_num_seqs=256), LLAMA3_8B, H100).summary()
    stat = simulate(
        w, EngineConfig(num_blocks=8000, policy="static", max_num_seqs=64), LLAMA3_8B, H100
    ).summary()
    assert cont["throughput_tok_s"] > 1.5 * stat["throughput_tok_s"]
    assert cont["ttft_p99_s"] < stat["ttft_p99_s"] / 10
    assert stat["completed"] == 150


def test_prefix_caching_cuts_prefill_and_ttft() -> None:
    w = shared_prefix_workload(n=100, rate=12, seed=5)
    on = simulate(w, EngineConfig(num_blocks=8000), LLAMA3_8B, H100).summary()
    off = simulate(w, EngineConfig(num_blocks=8000, prefix_caching=False), LLAMA3_8B, H100).summary()
    assert on["prefix_hit_rate"] > 0.8 and off["prefix_hit_rate"] == 0.0
    assert on["prefill_tokens_computed"] < 0.3 * off["prefill_tokens_computed"]
    assert on["ttft_p50_s"] < off["ttft_p50_s"]


def test_chunked_prefill_protects_decode_latency() -> None:
    w = poisson_workload(n=120, rate=8, prompt_mean=3000, output_mean=150, max_prompt=8192, seed=2)
    chunked = simulate(w, EngineConfig(num_blocks=12000, max_batched_tokens=512), LLAMA3_8B, H100).summary()
    plain = simulate(
        w, EngineConfig(num_blocks=12000, chunked_prefill=False, max_batched_tokens=8192), LLAMA3_8B, H100
    ).summary()
    assert chunked["tpot_p99_s"] < plain["tpot_p99_s"] / 2


def test_edge_cases() -> None:
    huge = Request(0, 0.0, tuple(range(5000)), (1, 2))
    one = Request(1, 0.5, tuple(range(10)), (7,))
    late = Request(2, 30.0, tuple(range(10)), (7, 8, 9))
    res = simulate([huge, one, late], EngineConfig(num_blocks=64), LLAMA3_8B, H100)
    assert res.rejected == (0,)
    assert [m.req_id for m in res.requests] == [1, 2]
    assert res.requests[0].tpot == 0.0
    assert SimResult((), (), 0.0, 0, 0, 0.0, 0, 0.0, 0.0).summary() == {"completed": 0.0}
    for bad in ({"policy": "fifo"}, {"num_blocks": 0}):
        with pytest.raises(ValueError):
            EngineConfig(**bad)
    nochunk_small = simulate(
        poisson_workload(n=20, rate=5, seed=9),
        EngineConfig(num_blocks=4000, chunked_prefill=False, max_batched_tokens=64),
        LLAMA3_8B,
        H100,
    )
    assert len(nochunk_small.requests) == 20  # long prompts still run alone when the batch is empty
    static_nochunk = simulate(
        poisson_workload(n=20, rate=5, seed=9),
        EngineConfig(num_blocks=4000, chunked_prefill=False, policy="static", max_num_seqs=8),
        LLAMA3_8B,
        H100,
    )
    assert len(static_nochunk.requests) == 20


def test_cli_and_api(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(cli_app, ["simulate", "--n", "40", "--rate", "10", "--num-blocks", "2000"])
    assert res.exit_code == 0, res.stdout
    assert json.loads(res.stdout)["completed"] == 40.0
    res = runner.invoke(
        cli_app,
        [
            "simulate",
            "--workload",
            "shared-prefix",
            "--n",
            "30",
            "--no-prefix-caching",
            "--no-chunked-prefill",
            "--max-batched-tokens",
            "4096",
        ],
    )
    assert json.loads(res.stdout)["prefix_hit_rate"] == 0.0
    res = runner.invoke(cli_app, ["simulate", "--workload", "multi-turn", "--n", "20", "--policy", "static"])
    assert res.exit_code == 0
    out = tmp_path / "r" / "bench.json"
    res = runner.invoke(cli_app, ["bench", "--quick", "--output", str(out)])
    assert res.exit_code == 0, res.stdout
    report = json.loads(out.read_text())
    assert report["kernel"]["bit_exact"] and len(report["batching"]) == 6
    assert "Automatic prefix caching" in res.stdout
    client = TestClient(create_app())
    assert client.get("/healthz").json()["version"] == __version__
    r = client.post(
        "/v1/simulate", json={"workload": "multi-turn", "n": 20, "rate": 2, "gpu": "a100", "num_blocks": 2000}
    ).json()
    assert r["completed"] == 20.0 and r["num_blocks"] == 2000
    assert client.post("/v1/simulate", json={"n": 0}).status_code == 422
