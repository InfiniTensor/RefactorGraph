#include "cuda_mem.h"
#include <cuda.h>

namespace refactor::kernel::cuda {

    void *malloc(size_t bytes) {
        void *ans;
        cudaMalloc(&ans, bytes);
        return ans;
    }
    void free(void *ptr) {
        cudaFree(ptr);
    }
    void *memcpy_h2d(void *dst, void const *src, size_t bytes) noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        return dst;
    }
    void *memcpy_d2h(void *dst, void const *src, size_t bytes) noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return dst;
    }
    void *memcpy_d2d(void *dst, void const *src, size_t bytes) noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        return dst;
    }

}// namespace refactor::kernel::cuda
