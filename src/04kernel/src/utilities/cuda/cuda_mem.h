#ifndef KERNEL_CUDA_MEM_H
#define KERNEL_CUDA_MEM_H

#include "mem_manager/mem_functions.h"

namespace refactor::kernel::cuda {

    void *malloc(size_t bytes) __attribute__((weak));
    void free(void *ptr) __attribute__((weak));
    void *memcpy_h2d(void *dst, void const *src, size_t bytes) noexcept __attribute__((weak));
    void *memcpy_d2h(void *dst, void const *src, size_t bytes) noexcept __attribute__((weak));
    void *memcpy_d2d(void *dst, void const *src, size_t bytes) noexcept __attribute__((weak));

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_MEM_H
