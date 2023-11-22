#include "kernel/cuda/split.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void splitKernel(
        unsigned long long n,
        uint8_t const *data, unsigned int const *segments, uint8_t **outputs,
        unsigned int outputCount,
        unsigned int sum,
        unsigned int sub) {
        extern __shared__ uint8_t *shared[];
        auto outputs_ = shared;
        auto segments_ = reinterpret_cast<unsigned int *>(shared + outputCount);
        for (auto i = threadIdx.x; i < outputCount; i += blockDim.x) {
            outputs_[i] = outputs[i];
            segments_[i] = segments[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto i = tid % sum, j = i * sub, k = 0u;
            while (j >= segments_[k]) { j -= segments_[k++]; }
            optimizedMemcpy(outputs_[k] + (tid / sum) * segments_[k] + j, data + tid * sub, sub);
        }
    }

    void launchSplit(
        KernelLaunchParameters const &params,
        void const *data, unsigned int const *segments, void **outputs,
        unsigned int outputCount,
        unsigned int sum,
        unsigned int sub) {
        splitKernel<<<
            params.gridSize,
            params.blockSize,
            outputCount *(sizeof(unsigned int) + sizeof(void *)),
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(data),
            segments,
            reinterpret_cast<uint8_t **>(outputs),
            outputCount,
            sum,
            sub);
    }

}// namespace refactor::kernel::cuda
