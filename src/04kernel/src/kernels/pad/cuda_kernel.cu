#include "cuda_kernel.hh"
#include "kernel/cuda/pad.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace refactor::kernel {
    using namespace runtime;

    auto PadCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        thrust::host_vector<cuda::DimInfo> dims(info.dims.size());
        std::transform(info.dims.begin(), info.dims.end(),
                       dims.begin(),
                       [](auto const &d) {
                           return cuda::DimInfo{
                               d.strideI,
                               d.strideO,
                               d.padS,
                               d.dimI,
                           };
                       });
        return [dims = thrust::device_vector<cuda::DimInfo>(dims),
                params = cuda::ThreadsDistributer()(info.blockCount),
                blockSize = info.blockSize,
                value = this->valueLength](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto src = reinterpret_cast<uint8_t const *>(inputs[0]);
            thrust::device_vector<uint8_t> defaultValue(blockSize, 0);
            if (value != 0) {
                auto constValue = reinterpret_cast<uint8_t const *>(inputs[2]);
                for (auto i : range0_(blockSize / value)) {
                    // std::memcpy(defaultValueHost.data() + i * value, constValue, value);
                    cudaMemcpy(defaultValue.data().get() + i * value, constValue, value, cudaMemcpyDeviceToDevice);
                }
            }
            cuda::launchPad(params, src, defaultValue.data().get(), dims.data().get(), outputs[0],
                            dims.size(),
                            blockSize);
        };
    }

}// namespace refactor::kernel

