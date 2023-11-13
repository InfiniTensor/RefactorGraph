#include "kernel/cuda/slice.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void sliceKernel(
        unsigned long long n,
        uint8_t const *src, DimInfo const *dims, uint8_t *output,
        unsigned int blockSize) {
    }

    void launchSlice(
        KernelLaunchParameters const &params,
        void const *src, DimInfo const *dims, void *output,
        unsigned int blockSize) {
        sliceKernel<<<
            params.gridSize,
            params.blockSize,
            params.dynamicSharedBytes,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(src),
            dims,
            reinterpret_cast<uint8_t *>(output),
            blockSize);
    }

}// namespace refactor::kernel::cuda
