#pragma once

#include "backend/cooperative_groups.cuh"
#include "backend/runtime_api.cuh"

#include <cstdint>

namespace sp1_gpu_backend {

template <typename T>
struct Plus {
    __device__ __forceinline__ T operator()(const T& lhs, const T& rhs) const {
        return lhs + rhs;
    }
};

// Shuffle a value as raw 32-bit limbs. This avoids backend implementations of
// cooperative_groups::plus<T>, which are not portable for SP1 field types.
template <typename Group, typename T, typename Op>
__device__ __forceinline__ T reduce(const Group& group, T value, Op op) {
    static_assert(sizeof(T) % sizeof(uint32_t) == 0);
    for (unsigned offset = group.size() / 2; offset > 0; offset >>= 1) {
        T other;
        auto* value_words = reinterpret_cast<uint32_t*>(&value);
        auto* other_words = reinterpret_cast<uint32_t*>(&other);
#pragma unroll
        for (unsigned i = 0; i < sizeof(T) / sizeof(uint32_t); ++i) {
            other_words[i] = __shfl_down_sync(0xffffffffu, value_words[i], offset, group.size());
        }
        if (group.thread_rank() + offset < group.size()) {
            value = op(value, other);
        }
    }
    return value;
}

} // namespace sp1_gpu_backend
