#include "kernel/cuda/where.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void whereKernel(
        unsigned long long n,
        unsigned int const *strides,
        bool const *c,
        uint8_t const *x,
        uint8_t const *y,
        uint8_t *output,
        unsigned int rank,
        unsigned int eleSize) {
        extern __shared__ unsigned int shared[];
        for (auto i = threadIdx.x; i < rank * 4; i += blockDim.x) {
            shared[i] = strides[i];
        }
        __syncthreads();
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            auto ic = 0u, ix = 0u, iy = 0u, rem = tid;
            for (auto j = 0u; j < rank; ++j) {
                auto dim = shared + 4 * j;
                auto quot = rem / dim[3];
                rem %= dim[3];
                ic += quot * dim[0];
                ix += quot * dim[1];
                iy += quot * dim[2];
            }

            optimizedMemcpy(output + tid * eleSize,
                            c[ic]
                                ? x + ix * eleSize
                                : y + iy * eleSize,
                            eleSize);
        }
    }

    void launchWhere(
        KernelLaunchParameters const &params,
        unsigned int const *strides,
        void const *c,
        void const *x,
        void const *y,
        void *output,
        unsigned int rank,
        unsigned int eleSize) {
        whereKernel<<<
            params.gridSize,
            params.blockSize,
            rank * sizeof(unsigned int) * 4,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            strides,
            reinterpret_cast<bool const *>(c),
            reinterpret_cast<uint8_t const *>(x),
            reinterpret_cast<uint8_t const *>(y),
            reinterpret_cast<uint8_t *>(output),
            rank,
            eleSize);
    }

}// namespace refactor::kernel::cuda
