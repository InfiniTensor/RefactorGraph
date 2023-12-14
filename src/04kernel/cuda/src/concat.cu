#include "kernel/cuda/concat.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void concatKernel(
        unsigned long long n,
        uint8_t const **inputs, unsigned int const *segments, uint8_t *output,
        unsigned int inputCount,
        unsigned int sum,
        unsigned int sub) {
        extern __shared__ uint8_t const *shared[];
        auto inputs_ = shared;
        auto segments_ = reinterpret_cast<unsigned int *>(shared + inputCount);
        for (auto i = threadIdx.x; i < inputCount; i += blockDim.x) {
            inputs_[i] = inputs[i];
            segments_[i] = segments[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto i = tid % sum, j = i * sub, k = 0u;
            while (j >= segments_[k]) { j -= segments_[k++]; }
            optimizedMemcpy(output + tid * sub, inputs_[k] + (tid / sum) * segments_[k] + j, sub);
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
            inputCount * (sizeof(unsigned int) + sizeof(void *)),
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
