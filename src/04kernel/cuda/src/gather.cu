#include "kernel/cuda/gather.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    template<class index_t>
    __global__ static void gatherKernel(
        unsigned long long n,
        uint8_t const *__restrict__ data,
        index_t const *__restrict__ indices,
        uint8_t *__restrict__ output,
        unsigned int batch,
        unsigned int unit,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto i = tid / batch,
                 j = tid % batch;
            auto index = __ldg(indices + i % midSizeO);
            optimizedMemcpy(unit * tid + output,
                            unit * (batch * (i / midSizeO * midSizeI + index) + j) + data,
                            unit);
        }
    }

    template<class index_t>
    void static launchGather(
        KernelLaunchParameters const &params,
        void const *data, void const *indices, void *output,
        unsigned int batch,
        unsigned int unit,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        gatherKernel<<<
            params.gridSize,
            params.blockSize,
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(data),
            reinterpret_cast<index_t const *>(indices),
            reinterpret_cast<uint8_t *>(output),
            batch,
            unit,
            midSizeI,
            midSizeO);
    }

    void launchGather(
        KernelLaunchParameters const &params,
        void const *data, void const *indices, void *output,
        bool i64,
        unsigned int batch,
        unsigned int unit,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        if (i64) {
            launchGather<int64_t>(
                params,
                data, indices, output,
                batch, unit,
                midSizeI, midSizeO);
        } else {
            launchGather<int32_t>(
                params,
                data, indices, output,
                batch, unit,
                midSizeI, midSizeO);
        }
    }

}// namespace refactor::kernel::cuda
