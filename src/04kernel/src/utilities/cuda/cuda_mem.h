#ifndef KERNEL_CUDA_MEM_H
#define KERNEL_CUDA_MEM_H

#include "mem_manager/mem_functions.h"

namespace refactor::kernel::cuda {

    void *malloc(size_t bytes);
    void free(void *ptr);
    void *memcpy_h2d(void *dst, void const *src, size_t bytes) noexcept;
    void *memcpy_d2h(void *dst, void const *src, size_t bytes) noexcept;
    void *memcpy_d2d(void *dst, void const *src, size_t bytes) noexcept;

    mem_manager::MemFunctions const &memFunc();

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_MEM_H
