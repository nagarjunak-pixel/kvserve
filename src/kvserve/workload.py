"""Synthetic serving workloads with realistic length distributions and prefix-sharing structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOCAB = 32_000


@dataclass(frozen=True, slots=True)
class Request:
    req_id: int
    arrival: float
    prompt: tuple[int, ...]
    output: tuple[int, ...]  # the tokens the model will emit (fixed so runs are reproducible)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt)

    @property
    def max_new_tokens(self) -> int:
        return len(self.output)


def _lognormal_lengths(
    rng: np.random.Generator, n: int, mean: float, sigma: float, lo: int, hi: int
) -> np.ndarray:
    mu = np.log(mean) - sigma**2 / 2  # so that E[X] == mean before clipping
    return np.clip(rng.lognormal(mu, sigma, size=n).round(), lo, hi).astype(int)


def _tokens(rng: np.random.Generator, n: int) -> tuple[int, ...]:
    return tuple(int(t) for t in rng.integers(1, VOCAB, size=n))


def poisson_workload(
    n: int = 200,
    rate: float = 4.0,
    prompt_mean: float = 512,
    output_mean: float = 128,
    seed: int = 0,
    max_prompt: int = 4096,
    max_output: int = 1024,
) -> list[Request]:
    """Independent requests: Poisson arrivals, log-normal prompt/output lengths (ShareGPT-like)."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate, size=n)
    arrivals = np.cumsum(gaps)
    plen = _lognormal_lengths(rng, n, prompt_mean, 0.8, 8, max_prompt)
    olen = _lognormal_lengths(rng, n, output_mean, 0.8, 1, max_output)
    return [
        Request(i, float(arrivals[i]), _tokens(rng, int(plen[i])), _tokens(rng, int(olen[i])))
        for i in range(n)
    ]


def shared_prefix_workload(
    n: int = 200,
    rate: float = 4.0,
    num_prefixes: int = 4,
    prefix_len: int = 1024,
    suffix_mean: float = 64,
    output_mean: float = 128,
    seed: int = 0,
) -> list[Request]:
    """Multi-tenant chat: every request starts with one of a few long shared system prompts."""
    rng = np.random.default_rng(seed)
    prefixes = [_tokens(rng, prefix_len) for _ in range(num_prefixes)]
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n))
    slen = _lognormal_lengths(rng, n, suffix_mean, 0.6, 4, 1024)
    olen = _lognormal_lengths(rng, n, output_mean, 0.8, 1, 1024)
    out: list[Request] = []
    for i in range(n):
        prefix = prefixes[int(rng.integers(num_prefixes))]
        out.append(
            Request(i, float(arrivals[i]), prefix + _tokens(rng, int(slen[i])), _tokens(rng, int(olen[i])))
        )
    return out


def multi_turn_workload(
    conversations: int = 40,
    turns: int = 5,
    rate: float = 2.0,
    system_len: int = 256,
    user_mean: float = 64,
    output_mean: float = 96,
    think_time: float = 3.0,
    seed: int = 0,
) -> list[Request]:
    """Agentic / chat sessions: turn k's prompt is turn k-1's prompt + its output + a new message."""
    rng = np.random.default_rng(seed)
    starts = np.cumsum(rng.exponential(1.0 / rate, size=conversations))
    out: list[Request] = []
    rid = 0
    for c in range(conversations):
        history = _tokens(rng, system_len)
        t = float(starts[c])
        for _ in range(turns):
            user = _tokens(rng, int(_lognormal_lengths(rng, 1, user_mean, 0.5, 4, 512)[0]))
            reply = _tokens(rng, int(_lognormal_lengths(rng, 1, output_mean, 0.6, 1, 512)[0]))
            prompt = history + user
            out.append(Request(rid, t, prompt, reply))
            rid += 1
            history = prompt + reply
            t += think_time + float(rng.exponential(think_time))
    out.sort(key=lambda r: (r.arrival, r.req_id))
    return [Request(i, r.arrival, r.prompt, r.output) for i, r in enumerate(out)]
