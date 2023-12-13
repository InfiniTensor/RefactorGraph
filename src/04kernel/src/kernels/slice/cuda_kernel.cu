#include "cuda_kernel.hh"
#include "kernel/cuda/slice.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto SliceCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        thrust::host_vector<cuda::DimInfo> dims(info.dims.size());
        std::transform(info.dims.begin(), info.dims.end(),
                       dims.begin(),
                       [](auto const &d) {
                           return cuda::DimInfo{
                               d.strideO,
                               d.skip,
                               d.strideI,
                           };
                       });
        return [dims = thrust::device_vector<cuda::DimInfo>(dims),
                params = cuda::ThreadsDistributer()(info.blockCount),
                blockSize = info.blockSize](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto src = reinterpret_cast<uint8_t const *>(inputs[0]);
            cuda::launchSlice(params, src, dims.data().get(), outputs[0],
                              dims.size(),
                              blockSize);
        };
    }

}// namespace refactor::kernel
