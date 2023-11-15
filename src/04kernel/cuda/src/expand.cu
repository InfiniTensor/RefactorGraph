#include "kernel/cuda/expand.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void expandKernel(
        unsigned long long n,
        uint8_t const *data, expand::DimStride const *strides, uint8_t *output,
        unsigned int rank,
        unsigned int eleSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid, i = 0;
            for (auto j = 0; j < rank; ++j) {
                auto const &s = strides[j];
                if (s.i) {
                    i += rem / s.o * s.i;
                }
                rem %= s.o;
            }

            memcpy(output + tid * eleSize, data + i * eleSize, eleSize);
        }
    }

    void launchExpand(
        KernelLaunchParameters const &params,
        void const *data, expand::DimStride const *strides, void *output,
        unsigned int rank,
        unsigned int eleSize) {
        expandKernel<<<
            params.gridSize,
            params.blockSize,
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            reinterpret_cast<uint8_t const *>(data),
            strides,
            reinterpret_cast<uint8_t *>(output),
            rank,
            eleSize);
    }

}// namespace refactor::kernel::cuda
