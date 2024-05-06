#ifndef KERNEL_CUDA_TOPK_CUH
#define KERNEL_CUDA_TOPK_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchTopK(
        KernelLaunchParameters const &params,
        float const *data,  float *dstVal, unsigned int  *dstIdx,       
        unsigned int topk,
        unsigned int stride_axis,
        unsigned int stride_in_pre,
        unsigned int stride_out_pre,
        unsigned int size_axis);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_TOPK_CUH
