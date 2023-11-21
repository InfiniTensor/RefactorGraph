#include "cuda_kernel.hh"
#include "kernel/cuda/concat.cuh"
#include "mem_manager/foreign_blob.hh"
#include "runtime/mem_manager.hh"
#include <thrust/device_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto ConcatCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        auto sub = std::min(info.submultiple(), 16u);
        auto workspaceSize = info.segments.size() * sizeof(void *);
        auto routine = [params = cuda::ThreadsDistributer()(info.blockCount * info.sum / sub),
                        segments = thrust::device_vector<dim_t>(info.segments.begin(), info.segments.end()),
                        workspaceSize,
                        sum = info.sum / sub,
                        sub](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            cudaMemcpy(workspace, inputs, workspaceSize, cudaMemcpyHostToDevice);
            cuda::launchConcat(
                params,
                reinterpret_cast<void const **>(workspace),
                segments.data().get(),
                outputs[0],
                segments.size(),
                sum,
                sub);
        };
        return RoutineWorkspace(std::move(routine), workspaceSize);
    }

}// namespace refactor::kernel
