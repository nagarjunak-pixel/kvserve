from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kvserve.blocks import BlockManager, OutOfBlocksError
from kvserve.cost import A100, H100, LLAMA3_8B, LLAMA3_70B, kv_capacity_blocks, step_cost
from kvserve.hashing import DEFAULT_SOURCE, BlockHasher, py_block_hashes

NATIVE = BlockHasher(4, use_native=True)
NUMPY = BlockHasher(4, use_native=False)

# ------------------------------------------------------------------------------- hashing


def test_native_kernel_builds() -> None:
    assert NATIVE.is_native, "clang++/g++ is required for the C++20 kernel"
    assert not NUMPY.is_native and NUMPY.backend == "numpy-fallback"
    with pytest.raises(ValueError):
        BlockHasher(0)


@given(tokens=st.lists(st.integers(0, 2**31 - 1), max_size=60), seed=st.integers(0, 2**64 - 1))
@settings(max_examples=80, deadline=None)
def test_hashes_bit_exact_and_chain_continuation(tokens: list[int], seed: int) -> None:
    a = BlockHasher(4, seed=seed).hashes(tokens)
    b = BlockHasher(4, seed=seed, use_native=False).hashes(tokens)
    assert np.array_equal(a, b)
    assert a.shape == (len(tokens) // 4,)
    if a.size >= 2:
        k = a.size // 2
        cont = NATIVE.hashes(tokens[k * 4 :], parent=int(BlockHasher(4, seed=seed).hashes(tokens)[k - 1]))
        assert np.array_equal(cont, a[k:])


def test_prefix_semantics() -> None:
    shared = list(range(100, 108))
    x = NATIVE.hashes(shared + [1, 2, 3, 4])
    y = NATIVE.hashes(shared + [9, 9, 9, 9])
    z = NATIVE.hashes([0] + shared[1:] + [1, 2, 3, 4])
    assert np.array_equal(x[:2], y[:2]) and x[2] != y[2]
    assert z[1] != x[1] and z[2] != x[2]  # a change in block 0 changes every later hash
    assert NATIVE.hashes([1, 2, 3]).size == 0
    assert py_block_hashes(np.zeros(0, dtype=np.int32), 4).size == 0


def test_batch_matches_individual() -> None:
    seqs = [list(range(10)), [], list(range(3)), list(range(50, 70))]
    batch = NATIVE.hashes_batch(seqs)
    for s, h in zip(seqs, batch, strict=True):
        assert np.array_equal(h, NATIVE.hashes(s))
    assert all(np.array_equal(a, b) for a, b in zip(NUMPY.hashes_batch(seqs), batch, strict=True))
    assert NATIVE.hashes_batch([]) == []


def test_hasher_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert not BlockHasher(source=tmp_path / "missing.cpp").is_native
    broken = tmp_path / "csrc" / "broken.cpp"
    broken.parent.mkdir()
    broken.write_text("??")
    assert not BlockHasher(source=broken).is_native
    fresh = tmp_path / "csrc" / "fresh.cpp"
    fresh.write_text(DEFAULT_SOURCE.read_text())
    assert BlockHasher(source=fresh).is_native
    monkeypatch.setenv("KVSERVE_DISABLE_NATIVE", "1")
    assert not BlockHasher().is_native
    monkeypatch.setenv("KVSERVE_DISABLE_NATIVE", "0")
    monkeypatch.setenv("PATH", "")
    assert not BlockHasher().is_native
    assert os.environ["PATH"] == ""


# ------------------------------------------------------------------------------- blocks


def _bm(num_blocks: int = 8, caching: bool = True) -> BlockManager:
    return BlockManager(num_blocks, 4, caching, BlockHasher(4))


def test_prefix_reuse_refcounts_and_lru_eviction() -> None:
    bm = _bm(6)
    p1 = list(range(10))  # 2 full blocks + 2 tokens
    assert bm.admit(1, p1) == 0
    bm.append_tokens(1, p1, p1)
    assert len(bm.seqs[1].blocks) == 3 and bm.used == 3
    p2 = list(range(8)) + [77, 78, 79]
    assert bm.match_prefix(p2) == bm.seqs[1].blocks[:2]
    assert bm.admit(2, p2) == 8
    assert bm.blocks[bm.seqs[1].blocks[0]].ref_count == 2
    bm.append_tokens(2, p2[8:], p2)
    bm.check_invariants()
    bm.free_seq(1)
    bm.free_seq(2)
    assert bm.used == 0 and len(bm.evictable) == 2 and len(bm.cache) == 2
    bm.check_invariants()
    # a re-admission hits the evictable cached blocks
    assert bm.admit(3, list(range(9))) == 8
    assert bm.hit_rate > 0
    bm.free_seq(3)
    # exhaust free blocks: evictable cached blocks are reclaimed in LRU order
    big = list(range(1000, 1024))
    bm.admit(4, big)
    bm.append_tokens(4, big, big)
    assert len(bm.cache) == len(
        {bm.blocks[b].block_hash for b in bm.seqs[4].blocks if bm.blocks[b].block_hash}
    )
    bm.check_invariants()


def test_out_of_blocks_leaves_state_unchanged() -> None:
    bm = _bm(2)
    p = list(range(8))
    bm.admit(1, p)
    bm.append_tokens(1, p, p)
    with pytest.raises(OutOfBlocksError):
        bm.append_tokens(1, [8], p + [8])
    assert bm.seqs[1].num_tokens == 8 and len(bm.seqs[1].blocks) == 2
    assert not bm.can_admit(list(range(50, 60)))
    bm.free_seq(1)
    with pytest.raises(OutOfBlocksError):
        bm2 = _bm(1)
        bm2._take()
        bm2._take()


def test_block_manager_validation_and_disabled_caching() -> None:
    with pytest.raises(ValueError):
        BlockManager(0, 4)
    with pytest.raises(ValueError):
        BlockManager(4, 4, hasher=BlockHasher(8))
    bm = _bm(8, caching=False)
    p = list(range(12))
    bm.admit(1, p)
    bm.append_tokens(1, p, p)
    assert bm.match_prefix(p) == [] and bm.admit(2, p) == 0
    bm.free_seq(1)
    assert len(bm.free) == 8 and bm.used == 0
    with pytest.raises(ValueError):
        bm.admit(2, p)
    assert bm.hit_rate == 0.0


def test_duplicate_content_blocks_are_not_double_cached() -> None:
    bm = _bm(8)
    p = list(range(8))
    for sid in (1, 2):  # both admitted before either computes, so neither hits the cache
        bm.admit(sid, p)
    for sid in (1, 2):
        bm.append_tokens(sid, p, p)
    assert len(bm.cache) == 2
    bm.free_seq(2)  # its blocks are duplicates -> straight back to the free list
    assert len(bm.evictable) == 0 and len(bm.free) == 6
    bm.free_seq(1)
    assert len(bm.evictable) == 2
    bm.check_invariants()


def test_invariant_checker_detects_corruption() -> None:
    bm = _bm(4)
    p = list(range(4))
    bm.admit(1, p)
    bm.append_tokens(1, p, p)
    bm.blocks[bm.seqs[1].blocks[0]].ref_count = 5
    with pytest.raises(AssertionError, match="refcount"):
        bm.check_invariants()
    bm.blocks[bm.seqs[1].blocks[0]].ref_count = 1
    bm.free.append(bm.seqs[1].blocks[0])
    with pytest.raises(AssertionError, match="states"):
        bm.check_invariants()
    bm.free.pop()
    bm.evictable[bm.free[-1]] = None
    with pytest.raises(AssertionError, match="both"):
        bm.check_invariants()


# --------------------------------------------------------------------------------- cost


def test_model_and_gpu_specs() -> None:
    assert LLAMA3_8B.kv_bytes_per_token == 2 * 32 * 8 * 128 * 2 == 131_072
    assert LLAMA3_8B.head_dim == 128
    assert kv_capacity_blocks(LLAMA3_8B, H100, 16) > 20_000
    assert kv_capacity_blocks(LLAMA3_8B, A100, 16) == kv_capacity_blocks(LLAMA3_8B, H100, 16)
    with pytest.raises(ValueError):
        kv_capacity_blocks(LLAMA3_70B, H100, 16)


def test_roofline_regimes() -> None:
    empty = step_cost(LLAMA3_8B, H100, [], [])
    assert empty.seconds == 0.0
    one = step_cost(LLAMA3_8B, H100, [], [512])
    many = step_cost(LLAMA3_8B, H100, [], [512] * 64)
    assert not one.compute_bound and many.seconds < 2 * one.seconds  # decode batching is ~free
    prefill = step_cost(LLAMA3_8B, H100, [(4096, 0)], [])
    assert prefill.compute_bound and prefill.seconds > 10 * one.seconds
    cached = step_cost(LLAMA3_8B, H100, [(64, 4032)], [])
    assert cached.seconds < prefill.seconds / 10  # prefix caching skips most prefill FLOPs
    assert many.bytes > one.bytes and prefill.flops > many.flops
