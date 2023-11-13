#ifndef KERNEL_CUDA_SLICE_CUH
#define KERNEL_CUDA_SLICE_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    struct DimInfo {
        unsigned int countStride, sizeStart;
        int sizeStride;
    };

    void launchSlice(
        KernelLaunchParameters const &,
        void const *src, DimInfo const *dims, void *output,
        unsigned int blockSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_SLICE_CUH
