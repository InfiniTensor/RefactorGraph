#include "cuda_kernel.hh"
#include "kernel/cuda/transpose.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {

    auto TransposeCuda::lower(Resources &) const noexcept -> Routine {
        thrust::host_vector<cuda::DimStride> strides(info.dims.size());
        std::transform(info.dims.begin(), info.dims.end(),
                       strides.begin(),
                       [](auto const &dim) { return cuda::DimStride{dim.strideI, dim.strideO}; });
        return [strides = thrust::device_vector<cuda::DimStride>(strides),
                params = cuda::ThreadsDistributer()(info.size),
                eleSize = dataType.size()](Resources &, void const **inputs, void **outputs) {
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
