#include "cuda_kernel.hh"
#include "kernel/cuda/expand.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto ExpandCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        using cuda::expand::DimStride;
        thrust::host_vector<DimStride> strides(info.strides.size());
        std::transform(info.strides.begin(), info.strides.end(),
                       strides.begin(),
                       [](auto const &s) { return DimStride{s.i, s.o}; });
        return [strides = thrust::device_vector<DimStride>(strides),
                params = cuda::ThreadsDistributer()(info.blockCount),
                eleSize = info.blockSize](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            cuda::launchExpand(
                params,
                inputs[0],
                strides.data().get(),
                outputs[0],
                strides.size(),
                eleSize);
        };
    }

}// namespace refactor::kernel
