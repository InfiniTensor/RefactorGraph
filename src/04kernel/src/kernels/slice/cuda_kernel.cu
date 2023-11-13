#include "cuda_kernel.hh"
#include "kernel/cuda/slice.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    Routine SliceCuda::lower(Resources &) const noexcept {
        auto reformed = info.reform(16);
        thrust::host_vector<cuda::DimInfo> dims(info.dims.size());
        std::transform(info.dims.begin(), info.dims.end(),
                       dims.begin(),
                       [](auto const &d) { return cuda::DimInfo{
                                               d.countStride,
                                               d.sizeStart,
                                               d.sizeStride,
                                           }; });
        return [dims = thrust::device_vector<cuda::DimInfo>(dims),
                params = cuda::ThreadsDistributer()(reformed.blockCount),
                blockSize = reformed.blockSize,
                baseOffset = reformed.baseOffset](Resources &, void const **inputs, void **outputs) {
            auto src = reinterpret_cast<uint8_t const *>(inputs[0]) + baseOffset;
            cuda::launchSlice(params, src, dims.data().get(), outputs[0],
                              dims.size(),
                              blockSize);
        };
    }

}// namespace refactor::kernel
