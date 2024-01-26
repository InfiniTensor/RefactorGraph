#include "kernel/cuda/pad.cuh"
#include "macro.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    __global__ static void padKernel(
        unsigned long long n,
        uint8_t const *__restrict__ src,
        uint8_t const *__restrict__ src_const,
        PadDimInfo const *__restrict__ dims,
        uint8_t *__restrict__ dst,
        unsigned int rank,
        unsigned int blockSize) {
        for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                  step = blockDim.x * gridDim.x;
             tid < n;
             tid += step) {
            long rem = tid, j = 0;
            bool flag = false;
            for (auto i = 0; i < rank; ++i) {
                auto strideO = __ldg(&(dims[i].strideO));
                auto strideI = __ldg(&(dims[i].strideI));
                auto padS = __ldg(&(dims[i].padS));
                auto dimI = __ldg(&(dims[i].dimI));
                auto pos = rem / strideO - padS;
                if (pos < 0 || pos >= dimI) {
                    flag = true;
                    break;
                }
                j += pos * strideI;
                rem %= strideO;
            }
            if (flag) {
                optimizedMemcpy(dst + tid * blockSize, src_const, blockSize);
            } else {
                optimizedMemcpy(dst + tid * blockSize, src + j * blockSize, blockSize);
            }
        }
    }

    void launchPad(
        KernelLaunchParameters const &params,
        uint8_t const *src, uint8_t const *src_const,
        PadDimInfo const *dims, void *output,
        unsigned int rank,
        unsigned int blockSize) {

        padKernel<<<
            params.gridSize,
            params.blockSize,
            0,
            reinterpret_cast<cudaStream_t>(params.stream)>>>(
            params.n,
            src,
            src_const,
            dims,
            reinterpret_cast<uint8_t *>(output),
            rank,
            blockSize);
    }

}// namespace refactor::kernel::cuda
