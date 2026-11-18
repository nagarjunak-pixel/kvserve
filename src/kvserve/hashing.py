"""Chained KV-block hashing (native C++20 with a bit-exact NumPy twin)."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

U64 = npt.NDArray[np.uint64]
I32 = npt.NDArray[np.int32]

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
DEFAULT_SEED = 0x6B76736572766521  # "kvserve!"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = _REPO_ROOT / "csrc" / "kvserve_kernels.cpp"
COMPILE_FLAGS = ["-O3", "-std=c++20", "-shared", "-fPIC"]
_u64p = ctypes.POINTER(ctypes.c_uint64)
_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)


def mix64(z: U64) -> U64:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
    out: U64 = z ^ (z >> np.uint64(31))
    return out


def py_block_hashes(tokens: I32, block_size: int, seed: int = DEFAULT_SEED) -> U64:
    """NumPy twin: within-block folds run vectorised across tokens, the chain runs per block."""
    blocks = tokens.shape[0] // block_size
    out = np.empty(blocks, dtype=np.uint64)
    if blocks == 0:
        return out
    grid = tokens[: blocks * block_size].astype(np.uint32).astype(np.uint64).reshape(blocks, block_size)
    with np.errstate(over="ignore"):
        grid = grid + GOLDEN
        parent = np.array([seed % 2**64], dtype=np.uint64)
        for k in range(blocks):
            h = mix64(parent + GOLDEN)
            for j in range(block_size):
                h = mix64(h ^ grid[k, j : j + 1])
            out[k] = h[0]
            parent = h
    return out


class BlockHasher:
    """Computes chained block hashes; ``use_native=False`` / ``KVSERVE_DISABLE_NATIVE=1`` forces NumPy."""

    def __init__(
        self,
        block_size: int = 16,
        seed: int = DEFAULT_SEED,
        source: Path | None = None,
        use_native: bool | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.block_size, self.seed = block_size, seed
        self._src = source or DEFAULT_SOURCE
        self._lib: ctypes.CDLL | None = None
        self.backend = "numpy-fallback"
        if use_native is None:
            use_native = os.environ.get("KVSERVE_DISABLE_NATIVE", "0") != "1"
        if use_native:
            self._load()

    @property
    def is_native(self) -> bool:
        return self._lib is not None

    def _load(self) -> None:
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None or not self._src.exists():
            return
        cache = self._src.parent.parent / ".kvserve_cache"
        cache.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(self._src.read_bytes() + " ".join(COMPILE_FLAGS).encode()).hexdigest()[:12]
        ext = "dylib" if platform.system() == "Darwin" else "so"
        lib_path = cache / f"libkvserve_{digest}.{ext}"
        if not lib_path.exists():
            tmp = cache / f"libkvserve_{digest}.{os.getpid()}.tmp"
            proc = subprocess.run(
                [compiler, *COMPILE_FLAGS, str(self._src), "-o", str(tmp)],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return
            tmp.replace(lib_path)
        lib = ctypes.CDLL(str(lib_path))
        lib.kv_block_hashes.argtypes = [_i32p, ctypes.c_int64, ctypes.c_int32, ctypes.c_uint64, _u64p]
        lib.kv_block_hashes.restype = None
        lib.kv_block_hashes_batch.argtypes = [
            _i32p, _i64p, ctypes.c_int64, ctypes.c_int32, ctypes.c_uint64, _i64p, _u64p,
        ]  # fmt: skip
        lib.kv_block_hashes_batch.restype = None
        self._lib = lib
        self.backend = f"c++20-native ({Path(compiler).name} -O3)"

    def hashes(self, tokens: Sequence[int] | I32, parent: int | None = None) -> U64:
        """Chained hashes of the full blocks of ``tokens``.

        ``parent`` continues an existing chain: ``hashes(t[k*B:], parent=h[k-1])`` equals
        ``hashes(t)[k:]``, so appending blocks never requires re-hashing the prefix.
        """
        seed = self.seed if parent is None else parent
        t = np.ascontiguousarray(tokens, dtype=np.int32)
        if self._lib is None:
            return py_block_hashes(t, self.block_size, seed)
        out = np.empty(t.shape[0] // self.block_size, dtype=np.uint64)
        if out.size:
            self._lib.kv_block_hashes(
                t.ctypes.data_as(_i32p), t.shape[0], self.block_size, seed % 2**64, out.ctypes.data_as(_u64p)
            )
        return out

    def hashes_batch(self, sequences: Sequence[Sequence[int]]) -> list[U64]:
        lengths = [len(s) for s in sequences]
        if self._lib is None or not sequences:
            return [self.hashes(s) for s in sequences]
        flat = np.ascontiguousarray(np.concatenate([np.asarray(s, dtype=np.int32) for s in sequences]))
        offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        out_offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
        np.cumsum([n // self.block_size for n in lengths], out=out_offsets[1:])
        out = np.empty(int(out_offsets[-1]), dtype=np.uint64)
        self._lib.kv_block_hashes_batch(
            flat.ctypes.data_as(_i32p) if flat.size else None,
            offsets.ctypes.data_as(_i64p),
            len(sequences),
            self.block_size,
            self.seed % 2**64,
            out_offsets.ctypes.data_as(_i64p),
            out.ctypes.data_as(_u64p) if out.size else None,
        )
        return [out[out_offsets[i] : out_offsets[i + 1]] for i in range(len(sequences))]
