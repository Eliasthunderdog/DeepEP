#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/layout/ep/workspace.cuh>

namespace deep_ep::ep {

// Called by producer block 0 after the data covered by this mailbox is ready:
// metadata/reset completion for grouped arrivals.
// Join the CTA only when the caller has not already synchronized those stores.
template <bool kJoinCTA = true>
__device__ __forceinline__ void publish_epilogue_ready(uint32_t* ready, int num_blocks) {
    if constexpr (kJoinCTA)
        __syncthreads();
    asm volatile("" ::: "memory");
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x)
        ptx::st_release_sys(ready + i, uint32_t{1});
}

// The workspace starts zeroed. Every consumer CTA acknowledges its own mailbox;
// a subsequent invocation on this workspace cannot start until this kernel exits.
// This also works across graph replays, without host-managed epochs or grid sync.
template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void wait_epilogue_ready(uint32_t* ready) {
    if (threadIdx.x == 0) {
        const auto slot = ready + blockIdx.x;
        comm::timeout_while<kNumTimeoutCycles>([=](bool last) {
            if (ptx::ld_acquire_sys(slot) == 1)
                return true;
            if (last)
                printf("DeepEP epilogue ready timeout: block=%u slot=%p\n", blockIdx.x, slot);
            return false;
        });
        asm volatile("" ::: "memory");
        ptx::st_relaxed_sys(slot, uint32_t{0});
    }
    __syncthreads();
    asm volatile("" ::: "memory");

    // Payload visibility is established separately by each group counter wait.
}

}  // namespace deep_ep::ep
