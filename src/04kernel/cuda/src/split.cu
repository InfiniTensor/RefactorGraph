#include "kernel/cuda/split.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void splitKernel(
        unsigned long long n,
        uint8_t const *data, unsigned int const *segments, uint8_t **outputs,
        unsigned int outputCount,
        unsigned int sum) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto offset = tid * sum;
            for (auto j = 0u; j < outputCount; ++j) {
                auto len = segments[j];
                memcpy(outputs[j] + tid * len, data + offset, len);
                offset += len;
            }
        }
    }

    void launchSplit(
        KernelLaunchParameters const &params,
        void const *data, unsigned int const *segments, void **outputs,
        unsigned int outputCount,
        unsigned int sum) {
        splitKernel<<<
            params.gridSize,
            params.blockSize,
            params.dynamicSharedBytes,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(data),
            segments,
            reinterpret_cast<uint8_t **>(outputs),
            outputCount,
            sum);
    }

}// namespace refactor::kernel::cuda
