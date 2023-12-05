#ifndef KERNEL_CUDA_FUNCTIONS_CUH
#define KERNEL_CUDA_FUNCTIONS_CUH

namespace refactor::kernel::cuda {

    void sync();
    void setCudaDevice(int);

    void copyOut(void *dst, const void *src, size_t size);

}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_FUNCTIONS_CUH
