#ifndef KERNEL_CUDA_EXPAND_CUH
#define KERNEL_CUDA_EXPAND_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    namespace expand {
        struct DimStride {
            unsigned int i, o;
        };
    }// namespace expand

    void launchExpand(
        KernelLaunchParameters const &,
        void const *data, expand::DimStride const *strides, void *output,
        unsigned int rank,
        unsigned int eleSize);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_EXPAND_CUH
