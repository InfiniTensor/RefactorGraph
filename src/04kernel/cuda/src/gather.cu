#include "kernel/cuda/gather.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    template<class index_t>
    __global__ void gatherKernel(
        unsigned long long n,
        uint8_t const *data,
        index_t const *indices,
        uint8_t *output,
        unsigned int batch,
        unsigned int unit,
        unsigned int midSizeI,
        unsigned int midSizeO) {
        extern __shared__ uint32_t shared[];
        for (auto i = threadIdx.x; i < midSizeO; i += blockDim.x) {
            shared[i] = indices[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto i = tid / batch,
                 j = tid % batch;
            memcpy(unit * tid + output,
                   unit * (batch * (i / midSizeO * midSizeI + shared[i % midSizeO]) + j) + data,
                   unit);
        }
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
            gatherKernel<<<
                params.gridSize,
                params.blockSize,
                midSizeO * sizeof(uint32_t),
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                reinterpret_cast<uint8_t const *>(data),
                reinterpret_cast<int64_t const *>(indices),
                reinterpret_cast<uint8_t *>(output),
                batch,
                unit,
                midSizeI,
                midSizeO);
        } else {
            gatherKernel<<<
                params.gridSize,
                params.blockSize,
                midSizeO * sizeof(uint32_t),
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                reinterpret_cast<uint8_t const *>(data),
                reinterpret_cast<int32_t const *>(indices),
                reinterpret_cast<uint8_t *>(output),
                batch,
                unit,
                midSizeI,
                midSizeO);
        }
    }

}// namespace refactor::kernel::cuda
