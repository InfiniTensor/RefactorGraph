#ifndef KERNEL_CUDA_CONCAT_CUH
#define KERNEL_CUDA_CONCAT_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchConcat(
        KernelLaunchParameters const &,
        void const **inputs, unsigned int const *segments, void *output,
        unsigned int inputCount,
        unsigned int sum,
        unsigned int sub);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_CONCAT_CUH
