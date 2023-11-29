#ifndef KERNEL_CUDA_MACRO_CUH
#define KERNEL_CUDA_MACRO_CUH

#include "common.h"
#include <cuda.h>

#define CUDA_ASSERT(STATUS)                                                 \
    if (auto status = (STATUS); status != cudaSuccess) {                    \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with {}", \
                                  cudaGetErrorString(status)));             \
    }

__device__ __forceinline__ void optimizedMemcpy(void *dst, void const *src, size_t size) {
#define ASSIGN(TY)                                                         \
    case sizeof(TY):                                                       \
        *reinterpret_cast<TY *>(dst) = *reinterpret_cast<TY const *>(src); \
        break
    switch (size) {
        ASSIGN(double4);
        ASSIGN(float4);
        ASSIGN(float2);
        ASSIGN(float1);
        ASSIGN(uchar2);
        ASSIGN(uchar1);
        default:
            memcpy(dst, src, size);
            break;
    }
#undef ASSIGN
}

#endif// KERNEL_CUDA_MACRO_CUH
