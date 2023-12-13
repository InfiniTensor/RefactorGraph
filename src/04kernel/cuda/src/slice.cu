#include "kernel/cuda/slice.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void sliceKernel(
        unsigned long long n,
        uint8_t const *src, DimInfo const *dims, uint8_t *dst,
        unsigned int rank,
        unsigned int blockSize) {
        extern __shared__ DimInfo dimInfo[];
        for (auto i = threadIdx.x; i < rank; i += blockDim.x) {
            dimInfo[i] = dims[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid, j = 0;
            for (auto i = 0; i < rank; ++i) {
                auto const &dim = dimInfo[i];
                j += rem / dim.strideO * dim.strideI + dim.skip;
                rem %= dim.strideO;
            }
            optimizedMemcpy(dst + tid * blockSize, src + j * blockSize, blockSize);
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
            rank * sizeof(DimInfo),
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(src),
            dims,
            reinterpret_cast<uint8_t *>(output),
            rank,
            blockSize);
    }

}// namespace refactor::kernel::cuda
