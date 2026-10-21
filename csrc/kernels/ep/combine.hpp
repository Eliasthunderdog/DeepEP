#pragma once

#include <algorithm>
#include <cstdint>
#include <format>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <nccl.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_ep/layout/ep/counter.cuh>

#include "../../runtime/jit.hpp"

namespace deep_ep::ep {

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* get_direct_combine_counted_counter_buffer(void* buffer,
                                                       const layout::TokenLayout& token_layout,
                                                       const int& num_max_tokens_per_rank,
                                                       const int& num_scaleup_ranks,
                                                       const bool& allow_multiple_reduction) {
    const auto num_tokens_in_layout = allow_multiple_reduction ? std::min(num_scaleup_ranks, token_layout.num_topk) : token_layout.num_topk;
    const auto recv_buffer_layout = layout::BufferLayout<false>(
        token_layout, num_tokens_in_layout, num_max_tokens_per_rank, buffer);
    const auto send_buffer_layout = layout::BufferLayout<false>(
        token_layout, 0, num_max_tokens_per_rank,
        recv_buffer_layout.get_buffer_end_ptr());
    return math::advance_ptr(
        buffer,
        math::align<int64_t>(
            math::ptr_diff(send_buffer_layout.get_buffer_end_ptr(), buffer),
            layout::CounterLayout::kCounterStrideBytes));
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            topk_idx_t* combined_topk_idx,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            const deep_jit::NoRefPtr& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_combined_tokens,
                            const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const bool& use_cft_counted,
                            uint32_t* counted_scaleup_le_ids,
                            const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    if (use_cft_counted) {
#if !((defined(CUDART_VERSION) && CUDART_VERSION >= 13030) || (defined(CUDA_VERSION) && CUDA_VERSION >= 13030))
        EP_HOST_ASSERT(false and
                       "CFT counted combine requires CUDA 13.3+ counted-write support");
#endif
        EP_HOST_ASSERT(num_scaleout_ranks == 1 and
                       "CFT counted combine is only supported for direct combine");
        EP_HOST_ASSERT(is_scaleup_nvlink and
                       "CFT counted combine does not support GIN/RDMA scaleup peers in this release");
        EP_HOST_ASSERT(at::cuda::getCurrentDeviceProperties()->major >= 10 and
                       "Counted direct combine requires SM100+");
        EP_HOST_ASSERT(not (use_expanded_layout and not allow_multiple_reduction) and
                       "Counted direct combine does not support expanded send-all mode");
        EP_HOST_ASSERT(counted_scaleup_le_ids != nullptr);
    }

    // Decide warps
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        num_warps = num_scaleup_warps + num_forward_warps;
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                       "Invalid combine SM count, please try to match your dispatch config");
    } else if (use_cft_counted) {
        // Counted fabric sends need one 64-byte completion mbarrier per warp
        // after the TMA staging area. Trim warps until both regions fit.
        while (num_warps > 0) {
            const auto tma_smem_bytes =
                static_cast<int64_t>(num_warps) * token_layout.get_num_bytes<true>();
            const auto counted_smem_bytes =
                math::align<int64_t>(tma_smem_bytes, ptx::kFabricMBarrierAlignBytes) +
                static_cast<int64_t>(num_warps) * sizeof(ptx::fabric_mbarrier);
            if (counted_smem_bytes <= num_smem_bytes)
                break;
            -- num_warps;
        }
        EP_HOST_ASSERT(num_warps > 0 and
                       "Insufficient shared memory for counted direct combine");
    }

    if (use_cft_counted) {
        void* counter_buffer = get_direct_combine_counted_counter_buffer(
            buffer, token_layout, num_max_tokens_per_rank, num_scaleup_ranks, allow_multiple_reduction);
        const auto counter_layout = layout::CounterLayout(counter_buffer, num_scaleup_ranks);
        CUDA_RUNTIME_CHECK(cudaMemsetAsync(counter_buffer, 0, counter_layout.get_num_bytes(), stream));
    }

    const auto num_threads = num_warps * 32;

    // Compile
    std::string header_name, func_name;
    if (num_scaleout_ranks == 1) {
        header_name = "combine";
        func_name = std::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                use_cft_counted, is_scaleup_nvlink,
                                use_expanded_layout, allow_multiple_reduction,
                                num_sms,
                                num_threads / 32,
                                num_scaleup_ranks * num_scaleout_ranks,
                                hidden,
                                num_max_tokens_per_rank,
                                num_experts,
                                num_topk,
                                num_qps, num_timeout_cycles);
    } else {
        header_name = "hybrid_combine";
        func_name = std::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>",
                                use_expanded_layout, allow_multiple_reduction,
                                num_sms,
                                num_scaleup_warps, num_forward_warps,
                                num_scaleout_ranks, num_scaleup_ranks,
                                hidden,
                                num_max_tokens_per_rank,
                                num_experts,
                                num_topk,
                                num_qps,
                                num_timeout_cycles);
    }
    const auto kernel = jit->compile("combine", std::format(R"(
#include <deep_ep/impls/ep/{}.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::{});
}}
)", header_name, func_name));

    // Launch
    const auto options = deep_jit::cuda::LaunchOptions {
        .stream = stream.stream(),
        .num_smem_bytes = num_smem_bytes,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(num_threads, 1, 1),
        .cluster_dim = dim3(2 - (num_sms % 2), 1, 1),
        .cooperative = true,
    };
    if (num_scaleout_ranks == 1) {
        jit->launch(
            kernel, options,
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights),
            src_metadata, combined_topk_idx, psum_num_recv_tokens_per_scaleup_rank,
            nccl_dev_comm, nccl_window,
            buffer, workspace,
            counted_scaleup_le_ids,
            scaleup_rank_idx,
            num_reduced_tokens, num_combined_tokens
        );
    } else {
        jit->launch(
            kernel, options,
            static_cast<nv_bfloat16*>(x), static_cast<float*>(topk_weights),
            src_metadata,
            psum_num_recv_tokens_per_scaleup_rank,
            token_metadata_at_forward,
            channel_linked_list,
            nccl_dev_comm, nccl_window,
            buffer, workspace,
            scaleout_rank_idx, scaleup_rank_idx,
            num_reduced_tokens
        );
    }

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;

    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Compile
    const auto kernel = jit->compile("combine_reduce_epilogue", std::format(R"(
#include <deep_ep/impls/ep/combine_reduce_epilogue.cuh>

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&deep_ep::ep::combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)", use_expanded_layout, allow_multiple_reduction,
        num_sms, num_threads / 32,
        num_scaleout_ranks, num_scaleup_ranks,
        hidden,
        num_max_tokens_per_rank,
        num_experts, num_topk));

    // Launch
    jit->launch(
        kernel, {
            .stream = stream.stream(),
            .num_smem_bytes = num_smem_bytes,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_threads, 1, 1),
            .enable_pdl = true,
        },
        static_cast<nv_bfloat16*>(combined_x),
        combined_topk_weights,
        combined_topk_idx,
        reduce_buffer,
        bias_0, bias_1,
        num_combined_tokens,
        scaleout_rank_idx, scaleup_rank_idx
    );
}

}  // namespace deep_ep::ep
