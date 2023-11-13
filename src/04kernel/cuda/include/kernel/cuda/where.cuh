#ifndef KERNEL_CUDA_WHERE_CUH
#define KERNEL_CUDA_WHERE_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchWhere(
        KernelLaunchParameters const &,
        unsigned int const *strides,
        void const *c,
        void const *x,
        void const *y,
        void *output,
        unsigned int rank,
        unsigned int eleSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_WHERE_CUH
