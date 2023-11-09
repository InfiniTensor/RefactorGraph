﻿#ifndef KERNEL_CUDA_SPLIT_CUH
#define KERNEL_CUDA_SPLIT_CUH

#include "threads_distributer.cuh"

namespace refactor::kernel::cuda {

    void launchSplit(
        KernelLaunchParameters const &params,
        void const *data, unsigned int const *segments, void **outputs,
        unsigned int outputCount,
        unsigned int sum);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_SPLIT_CUH
