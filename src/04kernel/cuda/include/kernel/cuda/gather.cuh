#ifndef KERNEL_CUDA_GATHER_CUH
#define KERNEL_CUDA_GATHER_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchGather(
        KernelLaunchParameters const &,
        void const *data, void const *indices, void *output,
        bool i64,
        unsigned int postfix,
        unsigned int midSizeI,
        unsigned int midSizeO);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_GATHER_CUH
