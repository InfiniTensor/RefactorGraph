#include "cuda_kernel.hh"
#include "kernel/cuda/scatter_nd.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto ScatterNDCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        return [strides = thrust::device_vector<dim_t>(info.strides.begin(), info.strides.end()),
                params = cuda::ThreadsDistributer()(info.prefix),
                blockCount = info.blockCount,
                blockSize = info.blockSize](Resources &, void *, void const *const *inputs, void *const *outputs) {
            cuda::launchScatterND(
                params,
                inputs[0],
                inputs[1],
                inputs[2],
                outputs[0],
                strides.data().get(),
                strides.size(),
                blockCount,
                blockSize);
        };
    }

}// namespace refactor::kernel
