#ifdef USE_CUDA

#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

namespace refactor::kernel::cuda {
    void *malloc(size_t bytes);
    void free(void *ptr);
    void *memcpy_h2d(void *dst, void const *src, size_t bytes) noexcept;
    void *memcpy_d2h(void *dst, void const *src, size_t bytes) noexcept;
    void *memcpy_d2d(void *dst, void const *src, size_t bytes) noexcept;
}// namespace refactor::kernel::cuda

#endif// CUDA_MEM_CUH

#endif// USE_CUDA
