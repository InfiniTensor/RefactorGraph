#include "functions.cuh"
#include "memory.cuh"

namespace refactor::hardware {
    using M = NvidiaMemory;

    void *M::malloc(size_t size) noexcept {
        void *ptr;
        CUDA_ASSERT(cudaMalloc(&ptr, size));
        return ptr;
    }
    void M::free(void *ptr) noexcept {
        if (auto status = cudaFree(ptr); status != cudaSuccess && status != cudaErrorCudartUnloading) {
            RUNTIME_ERROR(fmt::format("cudaFree failed with \"{}\" ({})",
                                      cudaGetErrorString(status), (int) status));
        }
    }
    void *M::copyHD(void *dst, void const *src, size_t bytes) const noexcept {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    }
    void *M::copyDH(void *dst, void const *src, size_t bytes) const noexcept {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    }
    void *M::copyDD(void *dst, void const *src, size_t bytes) const noexcept {
        CUDA_ASSERT(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
    }

}// namespace refactor::hardware
