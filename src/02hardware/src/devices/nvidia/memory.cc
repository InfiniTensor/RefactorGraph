#ifdef USE_CUDA

#include "memory.hh"
#include "hardware/devices/nvidia.h"
#include <cuda_runtime.h>

namespace refactor::hardware {
    using M = NvidiaMemory;

    void *M::malloc(size_t size) {
        void *ptr;
        CUDA_ASSERT(cudaMalloc(&ptr, size));
        return ptr;
    }
    void M::free(void *ptr) {
        if (auto status = cudaFree(ptr); status != cudaSuccess && status != cudaErrorCudartUnloading) {
            RUNTIME_ERROR(fmt::format("cudaFree failed with \"{}\" ({})",
                                      cudaGetErrorString(status), (int) status));
        }
    }
    void *M::copyHD(void *dst, void const *src, size_t bytes) const {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice))
        return dst;
    }
    void *M::copyDH(void *dst, void const *src, size_t bytes) const {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
        return dst;
    }
    void *M::copyDD(void *dst, void const *src, size_t bytes) const {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
        return dst;
    }

}// namespace refactor::hardware

#endif
