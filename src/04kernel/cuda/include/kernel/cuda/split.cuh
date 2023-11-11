#ifndef KERNEL_CUDA_SPLIT_CUH
#define KERNEL_CUDA_SPLIT_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchSplit(
        KernelLaunchParameters const &,
        void const *data, unsigned int const *segments, void **outputs,
        unsigned int outputCount,
        unsigned int sum,
        unsigned int sub);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_SPLIT_CUH
