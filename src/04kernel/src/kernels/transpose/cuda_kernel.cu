#include "cuda_kernel.hh"
#include "kernel/cuda/transpose.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {

    auto TransposeCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        using cuda::transpose::DimStride;
        thrust::host_vector<DimStride> strides(info.dims.size());
        std::transform(info.dims.begin(), info.dims.end(),
                       strides.begin(),
                       [](auto const &dim) { return DimStride{dim.strideI, dim.strideO}; });
        return [strides = thrust::device_vector<DimStride>(strides),
                params = cuda::ThreadsDistributer()(info.size),
                eleSize = dataType.size()](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            cuda::launchTranspose(
                params,
                inputs[0],
                strides.data().get(),
                outputs[0],
                strides.size(),
                eleSize);
        };
    }

}// namespace refactor::kernel
