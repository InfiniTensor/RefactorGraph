#ifndef KERNEL_CUDA_TRANSPOSE_CUH
#define KERNEL_CUDA_TRANSPOSE_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    namespace transpose {
        struct DimStride {
            unsigned int i, o;
        };
    }// namespace transpose

    void launchTranspose(
        KernelLaunchParameters const &,
        void const *data, transpose::DimStride const *strides, void *output,
        unsigned int rank,
        unsigned int eleSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_TRANSPOSE_CUH
