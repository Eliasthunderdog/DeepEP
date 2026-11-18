#pragma once

#include <deep_ep/comm/barrier.cuh>
#include <deep_ep/layout/ep/counter.cuh>

namespace deep_ep::ep {

// These kernels consume fabric destinations before grid completion, so PDL no
// longer supplies their visibility boundary. Explicitly cross the fabric,
// generic, and async proxies before reading payload/metadata or issuing TMA.
template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void wait_group_counted(const uint64_t* counter, uint64_t expected) {
    if (ptx::elect_one_sync()) {
        comm::timeout_while<kNumTimeoutCycles>([=](bool last) {
            const auto value = ptx::ld_acquire_sys(counter);
            if (value >= expected)
                return true;
            if (last)
                printf("DeepEP group counter timeout: counter=%p value=%llu expected=%llu\n",
                       counter, (unsigned long long)value, (unsigned long long)expected);
            return false;
        });
    }
    __syncwarp();
#if DEEP_EP_HAS_CFT_COUNTED_WRITES
    asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
#endif
    asm volatile("fence.proxy.async.global;" ::: "memory");
    __syncwarp();
}

}  // namespace deep_ep::ep
