#include "kernel/cuda/gather.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ void scatterNDKernel(
        size_t n,
        uint8_t *__restrict__ out,
        uint8_t const *__restrict__ in,
        int64_t const *__restrict__ indices,
        unsigned int const *__restrict__ strides,
        size_t rank,
        size_t blockSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            unsigned int j = 0;
            auto i = indices + tid * rank;
            for (auto k = 0; k < rank; ++k) {
                j += i[k] * __ldg(strides + k);
            }
            optimizedMemcpy(out + j * blockSize,
                            in + tid * blockSize,
                            blockSize);
        }
    }

    void launchScatterND(
        KernelLaunchParameters const &params,
        void const *data,
        void const *indices,
        void const *updates,
        void *output,
        unsigned int const *strides,
        size_t rank,
        unsigned int blockCount,
        size_t blockSize) {
        if (output != data) {
            cudaMemcpyAsync(
                output,
                data,
                blockCount * blockSize,
                cudaMemcpyDeviceToDevice);
        }
        scatterNDKernel<<<
            params.gridSize,
            params.blockSize,
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t *>(output),
            reinterpret_cast<uint8_t const *>(updates),
            reinterpret_cast<int64_t const *>(indices),
            strides,
            rank,
            blockSize);
    }

}// namespace refactor::kernel::cuda
