#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sstream>

#include <nccl.h>
#include <nccl_device/core.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/layout/bucket/workspace.cuh>

#include "api.hpp"
#include "../../runtime/jit.hpp"
#include "../../utils/system.hpp"


namespace deep_ep::comm {

#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
static void wait_logical_endpoint_ready(const CUlogicalEndpointId& le_id) {
    int ready = 0;
    const auto start = std::chrono::steady_clock::now();
    while (not ready) {
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointQuery(le_id, 1, &ready));
        if (ready)
            break;

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start).count() > 30)
            EP_HOST_UNREACHABLE("Timed out while waiting for CUDA logical endpoint readiness");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static CUlogicalEndpointProp make_counted_unicast_logical_endpoint_prop(const CUdevice& device,
                                                                        const int64_t& num_bytes) {
    CUlogicalEndpointProp prop = {};
    prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    prop.unicast.device = device;
    prop.size = static_cast<unsigned long long>(num_bytes);
    prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    prop.flags = CU_LOGICAL_ENDPOINT_FLAG_COUNTED_OPS;
    return prop;
}
#endif

pybind11::bytearray get_local_unique_id() {
    ncclUniqueId unique_id;
    NCCL_CHECK(ncclGetUniqueId(&unique_id));
    std::vector<char> result(sizeof(ncclUniqueId));
    std::memcpy(result.data(), &unique_id, sizeof(ncclUniqueId));
    return {result.data(), result.size()};
}

int64_t create_nccl_comm(const pybind11::bytearray& root_unique_id_bytes,
                         const int& num_ranks, const int& rank_idx) {
    // Copy unique ID
    ncclUniqueId root_unique_id;
    const auto root_unique_id_str = root_unique_id_bytes.cast<std::string>();
    std::memcpy(&root_unique_id, root_unique_id_str.c_str(), sizeof(ncclUniqueId));

    // Init
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, root_unique_id, rank_idx));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("New NCCL host communicator created (%d/%d)\n", rank_idx, num_ranks);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(const int64_t& nccl_comm) {
    NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(nccl_comm)));
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("NCCL host communicator aborted\n");
}

std::tuple<int, int> get_physical_domain_size(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};
}

std::tuple<int, int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}

std::tuple<int, int> get_gin_min_stride(const int64_t& nccl_comm) {
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks;

    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    EP_HOST_ASSERT(num_ranks % props.ginMinStride == 0);
    return {props.ginMinStride, num_ranks / props.ginMinStride};
}

Context::Context(const int64_t& nccl_comm, const symmetric::shared_comm_t& shared_comm,
                 const int& num_ranks, const int& rank_idx,
                 const int64_t& num_workspace_bytes,
                 const int64_t& num_gpu_buffer_bytes,
                 const int64_t& num_rdma_storage_bytes,
                 const bool& use_cpu_rdma_storage,
                 const bool& allow_hybrid_mode,
                 const std::optional<int>& sl_idx, const int& num_allocated_qps, const int& qp_depth,
                 const int& num_cpu_timeout_secs, const int& num_gpu_timeout_secs,
                 const bool& enable_lsa_multimem,
                 const std::shared_ptr<Context>& main_context):
    rank_idx(rank_idx), num_ranks(num_ranks),
    gin_min_stride(1),
    num_allocated_qps(num_allocated_qps),
    num_cpu_timeout_secs(num_cpu_timeout_secs),
    num_workspace_bytes(num_workspace_bytes),
    num_gpu_buffer_bytes(num_gpu_buffer_bytes),
    num_rdma_storage_bytes(num_rdma_storage_bytes),
    use_cpu_rdma_storage(use_cpu_rdma_storage) {
    EP_HOST_ASSERT(num_ranks > 0 and num_ranks <= kNumMaxRanks);
    EP_HOST_ASSERT(num_allocated_qps > 0 and num_allocated_qps <= kNumMaxQPs);
    EP_HOST_ASSERT(num_workspace_bytes > 0 and num_workspace_bytes % kNumAllocationAlignmentBytes == 0);
    EP_HOST_ASSERT(num_gpu_buffer_bytes >= 0 and num_gpu_buffer_bytes % kNumAllocationAlignmentBytes == 0);
    EP_HOST_ASSERT(num_rdma_storage_bytes >= 0 and num_rdma_storage_bytes % kNumAllocationAlignmentBytes == 0);

    num_gpu_timeout_cycles = static_cast<int64_t>(num_gpu_timeout_secs) * jit->device.get_clock_rate();
    int nccl_runtime_version;
    NCCL_CHECK(ncclGetVersion(&nccl_runtime_version));
    if (get_env("EP_BUFFER_DEBUG", 0)) {
        printf("DeepEP initialized with NCCL version: %d.%d.%d (loaded library)\n",
               nccl_runtime_version / 10000, (nccl_runtime_version % 10000) / 100, nccl_runtime_version % 100);
    }

    // Reuse the NCCL communicator
    comm = reinterpret_cast<ncclComm_t>(nccl_comm);

    // Print number of allocated QPs
    if (get_env<int>("EP_BUFFER_DEBUG"))
        printf("EP NCCL device communicator has %d allocated QPs\n", num_allocated_qps);

    // Query props
    ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCL_CHECK(ncclCommQueryProperties(comm, &props));
    EP_HOST_ASSERT((not enable_lsa_multimem or props.multimemSupport) and
                   "NCCL multimem is unavailable");

    // Initialize NCCL device communicator
    ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaMultimem = enable_lsa_multimem;
    if (get_env("EP_DISABLE_GIN", 0) == 0) {
        EP_HOST_ASSERT(
            (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE and
            "NCCL GIN is unavailable. This is usually due to a network configuration issue, "
            "such as `allow_hybrid_mode=0` (disable direct RDMA kernels) in multi-plane network.");
        EP_HOST_ASSERT(
            props.ginSupport[NCCL_GIN_TYPE_GDAKI] and
            "NCCL GDAKI is unavailable for this communicator.");

        gin_min_stride = props.ginMinStride;
        reqs.ginType = NCCL_GIN_TYPE_GDAKI;
        reqs.ginContextCount = num_allocated_qps;
        reqs.ginExclusiveContexts = true;
        reqs.ginQueueDepth = qp_depth > 0 ? qp_depth : kDefaultQPDepth;

        // Customized RDMA barrier needs extra signals
        reqs.ginSignalCount = num_ranks + 2 * 2;
        if (allow_hybrid_mode and num_rdma_storage_bytes > 0) {
            reqs.ginCustomStride = gin_min_stride;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_STRIDE;
        } else {
            reqs.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
        }

        // SL index
        if (std::getenv("EP_DEFAULT_RDMA_SL") != nullptr)
            reqs.ginTrafficClass = get_env<int>("EP_DEFAULT_RDMA_SL");
        if (sl_idx.has_value())
            reqs.ginTrafficClass = sl_idx.value();
        if (std::getenv("EP_OVERRIDE_RDMA_SL") != nullptr)
            reqs.ginTrafficClass = get_env<int>("EP_OVERRIDE_RDMA_SL");
    }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize);
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version and "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. Please re-compile DeepEP.");
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t));
#endif
    EP_HOST_ASSERT(dev_comm.ptr != nullptr);
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, static_cast<ncclDevComm_t*>(dev_comm.ptr)));

    // Now we know the NVLink domain size
    ncclTeam_t lsaTeam = ncclTeamLsa(comm);
    num_nvl_ranks = lsaTeam.nRanks, nvl_rank_idx = lsaTeam.rank;
    num_rdma_ranks = num_ranks / num_nvl_ranks, rdma_rank_idx = rank_idx / num_nvl_ranks;
    EP_HOST_ASSERT(num_rdma_ranks <= kNumMaxRDMARanks and num_nvl_ranks <= kNumMaxNVLRanks);
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0 and nvl_rank_idx == rank_idx % num_nvl_ranks);
    EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

    // Calculate scaleout/up domain size
    if (allow_hybrid_mode) {
        num_scaleout_ranks = num_rdma_ranks, num_scaleup_ranks = num_nvl_ranks;
        scaleout_rank_idx = rdma_rank_idx, scaleup_rank_idx = nvl_rank_idx;
    } else {
        num_scaleout_ranks = 1, num_scaleup_ranks = num_ranks;
        scaleout_rank_idx = 0, scaleup_rank_idx = rank_idx;
    }
    EP_HOST_ASSERT(num_scaleout_ranks <= kNumMaxScaleoutRanks and num_scaleup_ranks <= kNumMaxScaleupRanks);
    is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;

    if (main_context == nullptr) {
        symmetric_memory = symmetric::alloc(
            num_workspace_bytes + num_gpu_buffer_bytes, num_rdma_storage_bytes,
            use_cpu_rdma_storage,
            allow_hybrid_mode, gin_min_stride, rank_idx / gin_min_stride,
            shared_comm);
    } else {
        symmetric_memory = main_context->symmetric_memory;
    }

    // Create window
    // NOTES: `ncclCommWindowRegister` is collective: it internally calls bootstrapBarrier
    // across all ranks, so no explicit barrier is needed after this call.
    raw_window_ptr = this->symmetric_memory->ptr;
    NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, symmetric_memory->num_sym_bytes, &window, NCCL_WIN_STRICT_ORDERING));
    NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace));
    buffer = math::advance_ptr(workspace, num_workspace_bytes);
    if (enable_lsa_multimem)
        NCCL_CHECK(ncclGetLsaMultimemDevicePointer(window, num_workspace_bytes, &buffer_multimem));
    if (main_context == nullptr)
        CUDA_RUNTIME_CHECK(cudaMemset(workspace, 0, num_workspace_bytes));

    if (num_rdma_storage_bytes > 0) {
        auto* storage_base_ptr = math::advance_ptr(symmetric_memory->ptr, symmetric_memory->num_sym_bytes);
        local_rdma_storage_ptr = math::advance_ptr(storage_base_ptr, (rank_idx % gin_min_stride) * num_rdma_storage_bytes);

        rdma_storage_windows.resize(gin_min_stride, nullptr);
        for (int i = 0; i < gin_min_stride; ++ i) {
            auto* storage_ptr = math::advance_ptr(storage_base_ptr, i * num_rdma_storage_bytes);
            NCCL_CHECK(ncclCommWindowRegister(
                comm, storage_ptr, num_rdma_storage_bytes,
                &rdma_storage_windows[i], NCCL_WIN_GIN_ONLY));
        }

        CUDA_RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&rdma_storage_windows_gpu), sizeof(ncclWindow_t) * gin_min_stride));
        CUDA_RUNTIME_CHECK(cudaMemcpy(
            rdma_storage_windows_gpu, rdma_storage_windows.data(),
            sizeof(ncclWindow_t) * gin_min_stride, cudaMemcpyHostToDevice));
    }
}

void* Context::get_sym_ptr(void* ptr, const int& dst_nvl_rank_idx) const {
    EP_HOST_ASSERT(dst_nvl_rank_idx >= 0 and dst_nvl_rank_idx < num_nvl_ranks);
    void* dst_ptr = nullptr;
    const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(workspace);
    NCCL_CHECK(ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx, &dst_ptr));
    return dst_ptr;
}

std::tuple<int, int> Context::get_physical_domain_size() const {
    EP_HOST_ASSERT(not finalized);
    return {num_rdma_ranks, num_nvl_ranks};
}

std::tuple<int, int> Context::get_logical_domain_size() const {
    EP_HOST_ASSERT(not finalized);
    return {num_scaleout_ranks, num_scaleup_ranks};
}

bool Context::supports_counted_scaleup_le() const {
#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
    return num_ranks > 1 and num_ranks <= 32 and
           num_scaleout_ranks == 1 and
           is_scaleup_nvlink and
           num_nvl_ranks > 1;
#else
    return false;
#endif
}

bool Context::is_counted_scaleup_le_ready() const {
#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
    return counted_scaleup_le_ready;
#else
    return false;
#endif
}

std::vector<uint8_t> Context::export_counted_scaleup_le_handle() {
#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
    EP_HOST_ASSERT(supports_counted_scaleup_le());

    if (not counted_scaleup_le_created) {
        CUdevice device;
        CUDA_DRIVER_CHECK(lazy_cuCtxGetDevice(&device));

        auto prop = make_counted_unicast_logical_endpoint_prop(device, symmetric_memory->num_sym_bytes);
        cuuint64_t bind_alignment = 0, max_size = 0;
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointGetLimits(&bind_alignment, &max_size, &prop));
        EP_HOST_ASSERT(bind_alignment > 0 and
                       reinterpret_cast<uintptr_t>(raw_window_ptr) % bind_alignment == 0 and
                       static_cast<cuuint64_t>(symmetric_memory->num_sym_bytes) % bind_alignment == 0 and
                       static_cast<cuuint64_t>(symmetric_memory->num_sym_bytes) <= max_size);

        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointIdReserve(&counted_scaleup_le_id, 1));
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointCreate(counted_scaleup_le_id, &prop));
        wait_logical_endpoint_ready(counted_scaleup_le_id);
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointBindAddr(
            counted_scaleup_le_id, device, 0, raw_window_ptr, symmetric_memory->num_sym_bytes, 0));
        counted_scaleup_le_created = true;
    }

    CUlogicalEndpointFabricHandle fabric_handle = {};
    CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointExport(
        &fabric_handle, counted_scaleup_le_id, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));

    std::vector<uint8_t> handle(sizeof(CUlogicalEndpointFabricHandle));
    std::memcpy(handle.data(), &fabric_handle, handle.size());
    return handle;
#else
    return {};
#endif
}

void Context::import_counted_scaleup_le_handles(const std::vector<std::vector<uint8_t>>& handles,
                                                uint32_t* device_le_ids) {
#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
    EP_HOST_ASSERT(supports_counted_scaleup_le());
    EP_HOST_ASSERT(counted_scaleup_le_created);
    EP_HOST_ASSERT(not counted_scaleup_le_ready);
    EP_HOST_ASSERT(handles.size() == static_cast<size_t>(num_ranks));
    EP_HOST_ASSERT(num_scaleup_ranks == num_ranks);
    EP_HOST_ASSERT(device_le_ids != nullptr);

    std::vector<uint32_t> host_le_ids(num_scaleup_ranks, 0);
    const int local_lsa_rank_begin = rdma_rank_idx * num_nvl_ranks;
    const int local_lsa_rank_end = local_lsa_rank_begin + num_nvl_ranks;

    for (int rank = local_lsa_rank_begin; rank < local_lsa_rank_end; ++ rank) {
        EP_HOST_ASSERT(0 <= rank and rank < num_ranks);

        if (rank == rank_idx) {
            host_le_ids[rank] = counted_scaleup_le_id;
            continue;
        }

        EP_HOST_ASSERT(handles[rank].size() == sizeof(CUlogicalEndpointFabricHandle));
        CUlogicalEndpointFabricHandle fabric_handle = {};
        std::memcpy(&fabric_handle, handles[rank].data(), sizeof(CUlogicalEndpointFabricHandle));

        CUlogicalEndpointId imported_le_id = 0;
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointIdReserve(&imported_le_id, 1));
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointImport(
            imported_le_id, &fabric_handle, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
        wait_logical_endpoint_ready(imported_le_id);

        imported_counted_scaleup_le_ids.push_back(imported_le_id);
        host_le_ids[rank] = imported_le_id;
    }

    CUDA_RUNTIME_CHECK(cudaMemcpy(device_le_ids,
                                  host_le_ids.data(),
                                  host_le_ids.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
    counted_scaleup_le_ready = true;
#endif
}

void Context::finalize() {
    EP_HOST_ASSERT(not finalized);
    if (rdma_storage_windows_gpu != nullptr) {
        CUDA_RUNTIME_CHECK(cudaFree(rdma_storage_windows_gpu));
        rdma_storage_windows_gpu = nullptr;
    }
    for (const auto storage_window: rdma_storage_windows)
        NCCL_CHECK(ncclCommWindowDeregister(comm, storage_window));
    rdma_storage_windows.clear();
    local_rdma_storage_ptr = nullptr;

#if DEEP_EP_HAS_LOGICAL_ENDPOINTS
    for (auto it = imported_counted_scaleup_le_ids.rbegin();
         it != imported_counted_scaleup_le_ids.rend(); ++ it) {
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointDestroy(*it));
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointIdRelease(*it, 1));
    }
    imported_counted_scaleup_le_ids.clear();
    if (counted_scaleup_le_created) {
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointDestroy(counted_scaleup_le_id));
        CUDA_DRIVER_CHECK(lazy_cuLogicalEndpointIdRelease(counted_scaleup_le_id, 1));
        counted_scaleup_le_created = false;
        counted_scaleup_le_ready = false;
        counted_scaleup_le_id = 0;
    }
#endif

    // Deregister window
    NCCL_CHECK(ncclCommWindowDeregister(comm, window));
    symmetric_memory.reset();

    // Barrier can no longer be used
    barrier_signals = nullptr;

    // Destroy device communicator
    NCCL_CHECK(ncclDevCommDestroy(comm, static_cast<ncclDevComm_t*>(dev_comm.ptr)));
    free(dev_comm.ptr);
    finalized = true;
}

}  // namespace deep_ep::comm
