#include "kernel/cuda/gather.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    template<class index_t>
    __global__ void gatherKernel(
        unsigned long long n,
        uint8_t const *data,
        index_t const *indices,
        uint8_t *output,
        unsigned int postfix,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            memcpy(postfix * tid + output,
                   postfix * (tid / midSizeO * midSizeI + indices[tid % midSizeO]) + data,// NOTICE 先除后乘不能反，因为除要向下取整
                   postfix);
        }
    }

    void launchGather(
        KernelLaunchParameters const &params,
        void const *data, void const *indices, void *output,
        bool i64,
        unsigned int postfix,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        if (i64) {
            gatherKernel<<<
                params.gridSize,
                params.blockSize,
                params.dynamicSharedBytes,
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                reinterpret_cast<uint8_t const *>(data),
                reinterpret_cast<int64_t const *>(indices),
                reinterpret_cast<uint8_t *>(output),
                postfix,
                midSizeI,
                midSizeO);
        } else {
            gatherKernel<<<
                params.gridSize,
                params.blockSize,
                params.dynamicSharedBytes,
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                reinterpret_cast<uint8_t const *>(data),
                reinterpret_cast<int32_t const *>(indices),
                reinterpret_cast<uint8_t *>(output),
                postfix,
                midSizeI,
                midSizeO);
        }
    }

}// namespace refactor::kernel::cuda
