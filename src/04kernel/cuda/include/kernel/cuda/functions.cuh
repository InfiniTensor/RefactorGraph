#ifndef KERNEL_CUDA_FUNCTIONS_CUH
#define KERNEL_CUDA_FUNCTIONS_CUH

namespace refactor::kernel::cuda {

    int currentDevice();

    void sync();

    void copyOut(void *dst, const void *src, size_t size);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_FUNCTIONS_CUH
