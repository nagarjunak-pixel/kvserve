// Native C++20 (-O3) kernel for kvserve: chained KV-block hashing for automatic prefix caching.
//
// A KV block holding tokens t[k*B .. (k+1)*B) is identified by
//     h_k = fold(h_{k-1}, t[k*B .. (k+1)*B))      with h_{-1} = seed,
// so two requests share block k only if they share the entire prefix up to and including it
// (the vLLM automatic-prefix-caching invariant). The fold is SplitMix64-based and is reproduced
// exactly by the NumPy twin in kvserve/hashing.py.

#include <cstddef>
#include <cstdint>

namespace {

inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

}  // namespace

extern "C" {

// Hashes of the n / block_size *full* blocks of one token sequence.
void kv_block_hashes(
    const int32_t* __restrict__ tokens,
    int64_t n,
    int32_t block_size,
    uint64_t seed,
    uint64_t* __restrict__ out
) {
    const int64_t blocks = n / block_size;
    uint64_t parent = seed;
    for (int64_t k = 0; k < blocks; ++k) {
        uint64_t h = mix64(parent + kGolden);
        const int32_t* t = tokens + k * block_size;
        for (int32_t j = 0; j < block_size; ++j) {
            h = mix64(h ^ (static_cast<uint64_t>(static_cast<uint32_t>(t[j])) + kGolden));
        }
        out[k] = h;
        parent = h;
    }
}

// Batched variant over many sequences laid out back to back (offsets has n_seqs + 1 entries);
// out_offsets[i] is where sequence i's block hashes start in out.
void kv_block_hashes_batch(
    const int32_t* __restrict__ tokens,
    const int64_t* __restrict__ offsets,
    int64_t n_seqs,
    int32_t block_size,
    uint64_t seed,
    const int64_t* __restrict__ out_offsets,
    uint64_t* __restrict__ out
) {
    for (int64_t i = 0; i < n_seqs; ++i) {
        kv_block_hashes(tokens + offsets[i], offsets[i + 1] - offsets[i], block_size, seed, out + out_offsets[i]);
    }
}

}  // extern "C"
