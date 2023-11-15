#include "kernel/cuda/concat.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void concatKernel(
        unsigned long long n,
        uint8_t const **inputs, unsigned int const *segments, uint8_t *output,
        unsigned int inputCount,
        unsigned int sum,
        unsigned int sub) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto i = tid % sum, j = i * sub, k = 0u;
            while (j >= segments[k]) { j -= segments[k++]; }
            memcpy(output + tid * sub, inputs[k] + (tid / sum) * segments[k] + j, sub);
        }
    }

    void launchConcat(
        KernelLaunchParameters const &params,
        void const **inputs, unsigned int const *segments, void *output,
        unsigned int inputCount,
        unsigned int sum,
        unsigned int sub) {
        concatKernel<<<
            params.gridSize,
            params.blockSize,
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const **>(inputs),
            segments,
            reinterpret_cast<uint8_t *>(output),
            inputCount,
            sum,
            sub);
    }

}// namespace refactor::kernel::cuda
