"""PagedAttention block manager with reference counting and automatic prefix caching.

Physical KV memory is a pool of fixed-size blocks (Kwon et al., 2023). Each sequence owns a
block table. A *full* block is content-addressed by its chained hash, so a new request whose
prompt starts with an already-cached prefix reuses those physical blocks (refcount += 1) and
skips their prefill compute. Blocks whose refcount drops to zero stay cached and evictable in
LRU order until their memory is needed (vLLM v1 automatic prefix caching).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from kvserve.hashing import BlockHasher


@dataclass(slots=True)
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None  # set once the block is full and immutable


@dataclass(slots=True)
class SeqAlloc:
    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0  # tokens whose KV is materialised in the blocks
    cached_tokens: int = 0  # prefix tokens served from the prefix cache at admission


class OutOfBlocksError(RuntimeError):
    """Raised when an allocation cannot be satisfied even after evicting cached blocks."""


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = True,
        hasher: BlockHasher | None = None,
    ) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks, self.block_size = num_blocks, block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.hasher = hasher or BlockHasher(block_size)
        if self.hasher.block_size != block_size:
            raise ValueError("hasher block_size must match the block manager")
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free: list[int] = list(range(num_blocks - 1, -1, -1))  # never-used / invalidated blocks
        self.evictable: OrderedDict[int, None] = OrderedDict()  # refcount-0 cached blocks, LRU order
        self.cache: dict[int, int] = {}  # block hash -> block id
        self.seqs: dict[int, SeqAlloc] = {}
        self.hits = 0
        self.queries = 0

    # ------------------------------------------------------------------------------ accounting

    @property
    def num_free(self) -> int:
        """Blocks allocatable right now (truly free + evictable cached)."""
        return len(self.free) + len(self.evictable)

    @property
    def used(self) -> int:
        return self.num_blocks - self.num_free

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def _take(self) -> int:
        if self.free:
            return self.free.pop()
        if self.evictable:
            bid, _ = self.evictable.popitem(last=False)
            blk = self.blocks[bid]
            if blk.block_hash is not None and self.cache.get(blk.block_hash) == bid:
                del self.cache[blk.block_hash]
            blk.block_hash = None
            return bid
        raise OutOfBlocksError("KV cache exhausted")

    def _ref(self, bid: int) -> None:
        blk = self.blocks[bid]
        if blk.ref_count == 0:
            self.evictable.pop(bid, None)
        blk.ref_count += 1

    # -------------------------------------------------------------------------------- lifecycle

    def match_prefix(self, prompt: list[int]) -> list[int]:
        """Physical blocks holding the longest cached full-block prefix of ``prompt``.

        At least one prompt token is always left uncached so the model still produces logits.
        """
        if not self.enable_prefix_caching:
            return []
        hashes = self.hasher.hashes(prompt[: max(len(prompt) - 1, 0)])
        out: list[int] = []
        for h in hashes.tolist():
            bid = self.cache.get(h)
            if bid is None:
                break
            out.append(bid)
        return out

    def can_admit(self, prompt: list[int], reserve: int = 0) -> bool:
        cached = self.match_prefix(prompt)
        cached_evictable = sum(1 for b in cached if self.blocks[b].ref_count == 0)
        new_blocks = self.blocks_needed(len(prompt)) - len(cached)
        return self.num_free - cached_evictable >= new_blocks + reserve

    def admit(self, seq_id: int, prompt: list[int]) -> int:
        """Register a sequence; returns the number of prompt tokens served from the prefix cache."""
        if seq_id in self.seqs:
            raise ValueError(f"sequence {seq_id} already admitted")
        cached = self.match_prefix(prompt)
        self.queries += max(len(prompt) - 1, 0) // self.block_size if self.enable_prefix_caching else 0
        self.hits += len(cached)
        for bid in cached:
            self._ref(bid)
        alloc = SeqAlloc(blocks=list(cached), num_tokens=len(cached) * self.block_size)
        alloc.cached_tokens = alloc.num_tokens
        self.seqs[seq_id] = alloc
        return alloc.cached_tokens

    def append_tokens(self, seq_id: int, tokens: list[int], all_tokens: list[int]) -> None:
        """Materialise KV for ``tokens`` (the next tokens of ``all_tokens``), allocating blocks.

        Raises :class:`OutOfBlocksError` (leaving the sequence unchanged) if memory runs out.
        """
        alloc = self.seqs[seq_id]
        new_total = alloc.num_tokens + len(tokens)
        need = self.blocks_needed(new_total) - len(alloc.blocks)
        if need > self.num_free:
            raise OutOfBlocksError(f"need {need} blocks, {self.num_free} available")
        for _ in range(need):
            bid = self._take()
            self._ref(bid)
            alloc.blocks.append(bid)
        first_full = alloc.num_tokens // self.block_size
        alloc.num_tokens = new_total
        last_full = new_total // self.block_size
        if self.enable_prefix_caching and last_full > first_full:
            parent = self.blocks[alloc.blocks[first_full - 1]].block_hash if first_full else None
            chunk = all_tokens[first_full * self.block_size : last_full * self.block_size]
            for offset, h in enumerate(self.hasher.hashes(chunk, parent).tolist()):
                blk = self.blocks[alloc.blocks[first_full + offset]]
                blk.block_hash = h
                self.cache.setdefault(h, blk.block_id)

    def free_seq(self, seq_id: int) -> None:
        """Release a finished or preempted sequence; cached full blocks become evictable (LRU)."""
        alloc = self.seqs.pop(seq_id)
        for bid in reversed(alloc.blocks):  # tail blocks evicted first (least reusable)
            blk = self.blocks[bid]
            blk.ref_count -= 1
            if blk.ref_count < 0:  # pragma: no cover - guarded invariant
                raise AssertionError("negative refcount")
            if blk.ref_count == 0:
                if blk.block_hash is not None and self.cache.get(blk.block_hash) == bid:
                    self.evictable[bid] = None
                else:
                    blk.block_hash = None
                    self.free.append(bid)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0

    def check_invariants(self) -> None:
        """Every block is exactly one of: referenced, evictable-cached, or free."""
        refs = [0] * self.num_blocks
        for alloc in self.seqs.values():
            for bid in alloc.blocks:
                refs[bid] += 1
        for blk in self.blocks:
            if blk.ref_count != refs[blk.block_id]:
                raise AssertionError(
                    f"block {blk.block_id}: refcount {blk.ref_count} != {refs[blk.block_id]}"
                )
        free, evictable = set(self.free), set(self.evictable)
        if free & evictable:
            raise AssertionError("block both free and evictable")
        for blk in self.blocks:
            states = (blk.ref_count > 0) + (blk.block_id in free) + (blk.block_id in evictable)
            if states != 1:
                raise AssertionError(f"block {blk.block_id} is in {states} states")
