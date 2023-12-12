#ifndef KERNEL_CUDA_SCATTER_ND_CUH
#define KERNEL_CUDA_SCATTER_ND_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchScatterND(
        KernelLaunchParameters const &,
        void const *data,
        void const *indices,
        void const *updates,
        void *output,
        unsigned int const *strides,
        size_t rank,
        unsigned int blockCount,
        size_t blockSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_SCATTER_ND_CUH
