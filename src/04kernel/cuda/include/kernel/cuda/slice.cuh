#ifndef KERNEL_CUDA_SLICE_CUH
#define KERNEL_CUDA_SLICE_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    struct SliceDimInfo {
        unsigned int strideO, skip;
        int strideI;
    };

    void launchSlice(
        KernelLaunchParameters const &,
        void const *src, SliceDimInfo const *dims, void *output,
        unsigned int rank,
        unsigned int blockSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_SLICE_CUH
