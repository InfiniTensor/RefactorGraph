#include "kernel/cuda/transpose.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void transposeKernel(
        unsigned long long n,
        uint8_t const *__restrict__ data,
        transpose::DimStride const *__restrict__ strides,
        uint8_t *__restrict__ output,
        unsigned int rank,
        unsigned int eleSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto j = 0u, rem = tid;
            for (auto k = 0u; k < rank; ++k) {
                auto o_ = __ldg(&(strides[k].o));
                auto i_ = __ldg(&(strides[k].i));
                j += rem / o_ * i_;
                rem %= o_;
            }

            optimizedMemcpy(output + tid * eleSize, data + j * eleSize, eleSize);
        }
    }

    void launchTranspose(
        KernelLaunchParameters const &params,
        void const *data, transpose::DimStride const *strides, void *output,
        unsigned int rank,
        unsigned int eleSize) {
        transposeKernel<<<
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
