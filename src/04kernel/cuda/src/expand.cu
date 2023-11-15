#include "kernel/cuda/expand.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void expandKernel(
        unsigned long long n,
        uint8_t const *data, expand::DimStride const *strides, uint8_t *output,
        unsigned int rank,
        unsigned int eleSize) {
        extern __shared__ expand::DimStride shared[];
        for (auto i = threadIdx.x; i < rank; i += blockDim.x) {
            shared[i] = strides[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid, i = 0;
            for (auto j = 0; j < rank; ++j) {
                auto s = shared[j];
                i += rem / s.o * s.i;
                rem %= s.o;
            }
            optimizedMemcpy(output + tid * eleSize, data + i * eleSize, eleSize);
        }
    }

    void launchExpand(
        KernelLaunchParameters const &params,
        void const *data, expand::DimStride const *strides, void *output,
        unsigned int rank,
        unsigned int eleSize) {
        if (rank) {
            expandKernel<<<
                params.gridSize,
                params.blockSize,
                rank * sizeof(expand::DimStride),
                reinterpret_cast<cudaStream_t>(params.stream)>>>(
                params.n,
                reinterpret_cast<uint8_t const *>(data),
                strides,
                reinterpret_cast<uint8_t *>(output),
                rank,
                eleSize);
        } else if (data != output) {
            cudaMemcpyAsync(
                output,
                data,
                params.n * eleSize,
                cudaMemcpyDeviceToDevice,
                reinterpret_cast<cudaStream_t>(params.stream));
        }
    }

}// namespace refactor::kernel::cuda
