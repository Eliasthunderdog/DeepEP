#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/token.cuh>
#include <deep_ep/impls/ep/grouped_counted.cuh>
#include <deep_ep/impls/ep/epilogue_ready.cuh>

#include <deep_ep/impls/ep/combine_utils.cuh>

namespace deep_ep::ep {

template <bool kUseCFTCounted, bool kUseExpandedLayout, bool kAllowMultipleReduction,
          int kNumSMs, int kNumWarps,
          // TODO: merge these two variables into one (ensure the whole file does not contain "scaleup")
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int kHidden,
          int kNumMaxTokensPerRank,
          int kNumExperts, int kNumTopk,
          int64_t kNumTimeoutCycles,
          int kNumCounterGroups = 0,
          int kReduceWarpsPerToken = 1,
          int kNumThreads = kNumWarps * 32,
          int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16),
          int kNumRanks = kNumScaleoutRanks == 1 ? kNumScaleupRanks : kNumScaleoutRanks,
          bool kUseRankLayout = use_rank_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>(),
          int kNumTokensInLayout = get_num_tokens_in_layout<kAllowMultipleReduction, kNumRanks, kNumTopk>()>
__global__ void __launch_bounds__(kNumThreads, 1)
combine_reduce_epilogue_impl(nv_bfloat16* combined_x,
                             float* combined_topk_weights,
                             topk_idx_t* combined_topk_idx,
                             void* recv_buffer, void* workspace,
                             void* bias_0, void* bias_1,
                             const int num_combined_tokens,
                             const int scaleout_rank_idx, const int scaleup_rank_idx) {
    constexpr int kNumExpertsPerScaleout = kNumExperts / kNumScaleoutRanks;
    constexpr int kNumExpertsPerRank = kNumExperts / (kNumScaleupRanks * kNumScaleoutRanks);
    EP_STATIC_ASSERT(kNumExperts % (kNumScaleupRanks * kNumScaleoutRanks) == 0, "Invalid number of experts or ranks");

    EP_STATIC_ASSERT(kNumCounterGroups == (kUseCFTCounted ? 8 : 0), "Counted arrivals require eight groups");

    // Utils
    const auto sm_idx = static_cast<int>(blockIdx.x);
    const auto warp_idx = ptx::get_warp_idx(), lane_idx = ptx::get_lane_idx();
    EP_STATIC_ASSERT(kNumWarps % kReduceWarpsPerToken == 0, "Invalid reduction warp group");
    EP_STATIC_ASSERT(kReduceWarpsPerToken == 1 or kUseCFTCounted, "Cooperative reduction requires counted arrival");
    constexpr int kNumTokenWarps = kNumWarps / kReduceWarpsPerToken;
    const int token_warp_idx = warp_idx / kReduceWarpsPerToken;
    const int warp_in_token = warp_idx % kReduceWarpsPerToken;
    const auto global_warp_idx = token_warp_idx * kNumSMs + sm_idx;   // NOTES: Here we prioritize distributing tasks to different SMs to ensure that the last wave is evenly concentrated on each SM.

    // Load buffers from scale-out or scale-up ranks
    extern __shared__ __align__(kNumTMAAlignmentBytes) int8_t smem[];
    const auto comm_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, false);
    const auto comm_buffer = layout::BufferLayout<false>(
        comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);

    // Store buffers
    const auto output_token_layout = layout::TokenLayout(kNumHiddenBytes, 0, 0, false);
    const auto output_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, combined_x);
    const auto tma_buffer = layout::BufferLayout<false>(output_token_layout, kNumWarps, 1, smem)
        .get_rank_buffer(token_warp_idx).get_token_buffer(0);

    // Bias layout
    const auto bias_0_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_0);
    const auto bias_1_buffer = layout::BufferLayout<false>(output_token_layout, 1, num_combined_tokens, bias_1);

    if constexpr (kUseCFTCounted) {
        EP_STATIC_ASSERT(kNumScaleoutRanks == 1, "Counted reduce epilogue requires direct combine");
        EP_STATIC_ASSERT(kNumSMs <= layout::EPSignals::kNumMaxEpilogueBlocks, "Too many epilogue blocks");
        wait_epilogue_ready<kNumTimeoutCycles>(
            static_cast<layout::EPSignals*>(workspace)->combine_epilogue_ready);
    } else {
        cudaGridDependencySynchronize();
    }

    // Read from buffers and do reduction
    for (int token_idx = global_warp_idx; token_idx < num_combined_tokens; token_idx += kNumTokenWarps * kNumSMs) {
        // Preprocess all indices
        int stored_dst_rank_idx = -1, stored_dst_expert_idx = -1;
        EP_STATIC_ASSERT(kNumTopk <= 32, "Too many top-k selections");
        if (lane_idx < kNumTopk) {
            stored_dst_expert_idx = static_cast<int>(combined_topk_idx[token_idx * kNumTopk + lane_idx]);
            stored_dst_rank_idx = stored_dst_expert_idx >= 0 ?
                stored_dst_expert_idx / (kNumScaleoutRanks == 1 ? kNumExpertsPerRank : kNumExpertsPerScaleout) : -1;
        }
        __syncwarp();

        // Sort valid top-k indices to front
        const auto [should_deduplicate, deduplicate_key] = [&]() -> std::pair<bool, int> {
            if constexpr (kUseExpandedLayout and not kAllowMultipleReduction) {
                // Activations are never reduced before
                return {false, 0};
            } else if constexpr (kNumScaleoutRanks != 1 and not kUseExpandedLayout and not kAllowMultipleReduction) {
                // Hybrid mode without expanded layout and multiple reduction. Should deduplicate on a per-rank basis
                return {true, stored_dst_expert_idx >= 0 ? stored_dst_expert_idx / kNumExpertsPerRank : -1};
            } else {
                // Should deduplicate on a per-rank (for non-hybrid mode) or a per-scale-rank (for hybrid mode) basis
                return {true, stored_dst_rank_idx};
            }
        }();
        auto reduce_valid_mask = should_deduplicate ?
            ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
            ptx::gather(stored_dst_rank_idx >= 0);
        int topk_slot_idx[kNumTokensInLayout];
        compute_topk_slots(
            topk_slot_idx, reduce_valid_mask,
            [=](const int& idx) {
                return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx;
            }
        );

        if constexpr (kNumCounterGroups > 0) {
            const auto counter_base = math::advance_ptr(recv_buffer, math::align<int64_t>(
                math::ptr_diff(comm_buffer.get_buffer_end_ptr(), recv_buffer), layout::CounterLayout::kCounterStrideBytes));
            const auto groups = layout::GroupedCounterLayout<kNumCounterGroups, kNumMaxTokensPerRank>(counter_base, kNumRanks);
            const auto start = groups.group(token_idx) * groups.kGroupWidth;
            const auto end = min(start + groups.kGroupWidth, num_combined_tokens);
            // Each source rank updates a different counter. Lanes poll those
            // counters concurrently; there is no CTA or grid barrier here.
            uint64_t selected_ranks = 0;
            if constexpr (decltype(groups)::kGroupWidth == 1) {
                #pragma unroll
                for (int k = 0; k < kNumTopk; ++k) {
                    const int src = ptx::exchange(stored_dst_rank_idx, k);
                    if (src >= 0)
                        selected_ranks |= uint64_t{1} << src;
                }
            }
            int parallel_contributions[math::constexpr_ceil_div(kNumRanks, 32)] = {};
            if constexpr (decltype(groups)::kGroupWidth >= 8) {
                // Scan tokens in parallel, then count membership for each
                // source rank. Avoid a serial full-group scan in every lane.
                for (int base = start; base < end; base += 32) {
                    uint64_t token_ranks = 0;
                    const int t = base + lane_idx;
                    if (t < end) {
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++k) {
                            const int expert = combined_topk_idx[t * kNumTopk + k];
                            if (expert >= 0)
                                token_ranks |= uint64_t{1} << (expert / kNumExpertsPerRank);
                        }
                    }
                    #pragma unroll
                    for (int src = 0; src < kNumRanks; ++src) {
                        const int count = __popc(__ballot_sync(0xffffffff, (token_ranks >> src) & 1));
                        if (lane_idx == src % 32)
                            parallel_contributions[src / 32] += count;
                    }
                }
            }
            for (int src = lane_idx; src < kNumRanks; src += 32) {
                int contributions = 0;
                if constexpr (decltype(groups)::kGroupWidth == 1) {
                    contributions = (selected_ranks >> src) & 1;
                } else if constexpr (decltype(groups)::kGroupWidth >= 8) {
                    contributions = parallel_contributions[src / 32];
                } else {
                    for (int t = start; t < end; ++t) {
                        bool selected = false;
                        #pragma unroll
                        for (int k = 0; k < kNumTopk; ++k) {
                            const int expert = combined_topk_idx[t * kNumTopk + k];
                            selected |= expert >= 0 and expert / kNumExpertsPerRank == src;
                        }
                        contributions += selected;
                    }
                }
                if (contributions != 0) {
                    const auto counter = groups.get_counter_ptr(src, token_idx);
                    const auto expected = static_cast<uint64_t>(contributions) * comm_token_layout.get_num_bytes<false>();
                    comm::timeout_while<kNumTimeoutCycles>([=](bool last) {
                        const auto value = ptx::ld_acquire_sys(counter);
                        if (value >= expected)
                            return true;
                        if (last)
                            printf("DeepEP grouped rank counter timeout: rank=%d value=%llu expected=%llu\n",
                                   src, (unsigned long long)value, (unsigned long long)expected);
                        return false;
                    });
                }
            }
            __syncwarp();
#if DEEP_EP_HAS_CFT_COUNTED_WRITES
            asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
#endif
            asm volatile("fence.proxy.async.global;" ::: "memory");
            __syncwarp();
        }

        // Iterate over per-hidden-chunk stage
        using combine_vec_t = typename CombineVecTraits<kHidden * sizeof(nv_bfloat16)>::vec_t;
        constexpr int kHiddenVecTotal = kHidden * sizeof(nv_bfloat16) / sizeof(combine_vec_t);
        EP_STATIC_ASSERT(kHiddenVecTotal % (32 * kReduceWarpsPerToken) == 0, "Invalid cooperative hidden partition");
        constexpr int kHiddenVec = kHiddenVecTotal / kReduceWarpsPerToken;
        const int hidden_offset = warp_in_token * kHiddenVec;
        constexpr int kUnrollFactor = get_max_unroll_factor<kHiddenVec, 4>();
        combine_reduce<kHiddenVec, kUnrollFactor, kNumTokensInLayout, kUseCFTCounted>(
            lane_idx, topk_slot_idx, static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()) + hidden_offset,
            /* Get source base */ [=](const int& slot_idx) {
                return static_cast<combine_vec_t*>(
                    comm_buffer.get_rank_buffer(slot_idx).get_token_buffer(token_idx).get_base_ptr()) + hidden_offset;
            },
            /* Wait buffer release */ [=]() {
                ptx::tma_store_wait();
                __syncwarp();
            },
            /* Bias 0 */ bias_0 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_0_buffer.get_token_buffer(token_idx).get_base_ptr()) + hidden_offset,
            /* Bias 1 */ bias_1 == nullptr ?
                nullptr : static_cast<combine_vec_t*>(bias_1_buffer.get_token_buffer(token_idx).get_base_ptr()) + hidden_offset
        );
        ptx::tma_store_fence();
        __syncwarp();
        // Each warp owns disjoint input/output slices and its own TMA group.
        // It can issue and drain independently: no cross-warp barrier is needed.
        if (ptx::elect_one_sync()) {
            ptx::tma_store_1d(static_cast<combine_vec_t*>(output_buffer.get_token_buffer(token_idx).get_base_ptr()) + hidden_offset,
                              static_cast<combine_vec_t*>(tma_buffer.get_base_ptr()) + hidden_offset,
                              kNumHiddenBytes / kReduceWarpsPerToken);
            ptx::tma_store_commit();
        }
        __syncwarp();

        // Write top-k weights
        if (combined_topk_weights != nullptr and warp_in_token == 0) {
            const auto master_lane_idx = ptx::get_master_lane_idx(ptx::match(stored_dst_rank_idx));
            if (lane_idx < kNumTopk) {
                float value = 0;
                if (stored_dst_rank_idx >= 0) {
                    const auto dst_ptr = comm_buffer
                        .get_rank_buffer(kUseRankLayout ? stored_dst_rank_idx : master_lane_idx)
                        .get_token_buffer(token_idx).get_topk_weights_ptr() + lane_idx;
                    value = *dst_ptr;
                }
                combined_topk_weights[token_idx * kNumTopk + lane_idx] = value;
            }
            __syncwarp();
        }
    }

    // Inputs are acquired through the mailbox and, for grouped arrivals, counters.
    // Join the producer after reduction before its buffers can be reused.
    if constexpr (kUseCFTCounted)
        cudaGridDependencySynchronize();
}

}  // namespace deep_ep::ep
