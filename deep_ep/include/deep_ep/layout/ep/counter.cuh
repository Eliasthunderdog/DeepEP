#pragma once

#include <deep_ep/common/math.cuh>
#include <deep_ep/common/exception.cuh>

namespace deep_ep::layout {

struct CounterLayout {
    void* base;
    int num_scaleup_ranks;

    static constexpr int64_t kCounterStrideBytes = 256;

    __forceinline__ __device__ __host__
    CounterLayout(void* base,
                  const int& num_scaleup_ranks):
        base(base),
        num_scaleup_ranks(num_scaleup_ranks) {
        EP_UNIFIED_ASSERT(num_scaleup_ranks > 0);
    }

    __forceinline__ __device__ __host__
    static int64_t get_num_bytes(const int& num_scaleup_ranks) {
        EP_UNIFIED_ASSERT(num_scaleup_ranks > 0);
        return static_cast<int64_t>(num_scaleup_ranks) * kCounterStrideBytes;
    }

    __forceinline__ __device__ __host__
    int64_t get_num_bytes() const {
        return get_num_bytes(num_scaleup_ranks);
    }

    __forceinline__ __device__ __host__
    int64_t get_counter_offset(const int& src_scaleup_rank_idx) const {
        EP_UNIFIED_ASSERT(0 <= src_scaleup_rank_idx and src_scaleup_rank_idx < num_scaleup_ranks);
        return static_cast<int64_t>(src_scaleup_rank_idx) * kCounterStrideBytes;
    }

    __forceinline__ __device__ __host__
    uint64_t* get_counter_ptr(const int& src_scaleup_rank_idx) const {
        return math::advance_ptr<uint64_t>(
            base, get_counter_offset(src_scaleup_rank_idx));
    }

    __forceinline__ __device__ __host__
    void* get_buffer_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }
};

}  // namespace deep_ep::layout
