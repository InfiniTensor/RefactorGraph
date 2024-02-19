#ifndef KERNEL_CUDA_PAD_CUH
#define KERNEL_CUDA_PAD_CUH

#include "threads_distributer.cuh"
#include <cstdint>

namespace refactor::kernel::cuda {

    struct PadDimInfo {
        unsigned int strideI, strideO, padS, dimI;
    };

    void launchPad(
        KernelLaunchParameters const &,
        uint8_t const *src, uint8_t const *src_const,
        PadDimInfo const *dims, void *output,
        unsigned int rank,
        unsigned int blockSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_PAD_CUH
