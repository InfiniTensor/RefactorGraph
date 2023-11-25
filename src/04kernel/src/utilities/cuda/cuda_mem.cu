#include "cuda_mem.cuh"
#include <cuda.h>

namespace refactor::kernel::cuda {

    Arc<hardware::MemManager> BasicCudaMemManager::instance() {
        static auto I = std::make_shared<BasicCudaMemManager>();
        return I;
    }
    void *BasicCudaMemManager::malloc(size_t bytes) noexcept {
        void *ans;
        cudaMalloc(&ans, bytes);
        return ans;
    }
    void BasicCudaMemManager::free(void *ptr) noexcept {
        cudaFree(ptr);
    }
    void *BasicCudaMemManager::copyHD(void *dst, void const *src, size_t bytes) const noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        return dst;
    }
    void *BasicCudaMemManager::copyDH(void *dst, void const *src, size_t bytes) const noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return dst;
    }
    void *BasicCudaMemManager::copyDD(void *dst, void const *src, size_t bytes) const noexcept {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        return dst;
    }

}// namespace refactor::kernel::cuda
