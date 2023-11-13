#include "kernel/cuda/slice.cuh"
#include <cstdint>
#include <cstdio>

namespace refactor::kernel::cuda {

    __global__ static void sliceKernel(
        unsigned long long n,
        uint8_t const *src, DimInfo const *dims, uint8_t *output,
        unsigned int rank,
        unsigned int blockSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid;
            auto src_ = src;
            auto dst_ = output + rem * blockSize;
            for (auto i = 0; i < rank; ++i) {
                auto const &dim = dims[i];
                src_ += rem / dim.countStride * dim.sizeStride + dim.sizeStart;
                rem %= dim.countStride;
            }
            memcpy(dst_, src_, blockSize);
        }
    }

    void launchSlice(
        KernelLaunchParameters const &params,
        void const *src, DimInfo const *dims, void *output,
        unsigned int rank,
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
            rank,
            blockSize);
    }

}// namespace refactor::kernel::cuda
