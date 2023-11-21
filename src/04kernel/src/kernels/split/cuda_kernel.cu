#include "cuda_kernel.hh"
#include "kernel/cuda/split.cuh"
#include "mem_manager/foreign_blob.hh"
#include "runtime/mem_manager.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto SplitCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        auto sub = std::min(info.submultiple(), 16u);
        auto workspaceSize = info.segments.size() * sizeof(void *);
        auto routine = [params = cuda::ThreadsDistributer()(info.blockCount * info.sum / sub),
                        segments = thrust::device_vector<dim_t>(info.segments.begin(), info.segments.end()),
                        workspaceSize,
                        sum = info.sum / sub,
                        sub](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            cudaMemcpy(workspace, outputs, workspaceSize, cudaMemcpyHostToDevice);
            cuda::launchSplit(
                params,
                inputs[0],
                segments.data().get(),
                reinterpret_cast<void **>(workspace),
                segments.size(),
                sum,
                sub);
        };
        return RoutineWorkspace(std::move(routine), workspaceSize);
    }

}// namespace refactor::kernel
