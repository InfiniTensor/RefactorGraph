#ifndef KERNEL_CUDA_MACRO_CUH
#define KERNEL_CUDA_MACRO_CUH

#include "common.h"
#include <cuda.h>

#define CUDA_ASSERT(STATUS)                                                 \
    if (auto status = (STATUS); status != cudaSuccess) {                    \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with {}", \
                                  cudaGetErrorString(status)));             \
    }

#endif// KERNEL_CUDA_MACRO_CUH
