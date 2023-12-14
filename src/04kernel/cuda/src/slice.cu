#include "kernel/cuda/slice.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void sliceKernel(
        unsigned long long n,
        uint8_t const *__restrict__ src,
        DimInfo const *__restrict__ dims,
        uint8_t *__restrict__ dst,
        unsigned int rank,
        unsigned int blockSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid, j = 0;
            for (auto i = 0; i < rank; ++i) {
                auto strideO = __ldg(&(dims[i].strideO));
                auto strideI = __ldg(&(dims[i].strideI));
                auto skip = __ldg(&(dims[i].skip));
                j += rem / strideO * strideI + skip;
                rem %= strideO;
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
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(src),
            dims,
            reinterpret_cast<uint8_t *>(output),
            rank,
            blockSize);
    }

}// namespace refactor::kernel::cuda
