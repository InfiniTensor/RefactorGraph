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

    mem_manager::MemFunctions const &memFunc() {
        static mem_manager::MemFunctions F{
            malloc,
            free,
            memcpy_h2d,
            memcpy_d2h,
            memcpy_d2d,
        };
        return F;
    }


}// namespace refactor::kernel::cuda
