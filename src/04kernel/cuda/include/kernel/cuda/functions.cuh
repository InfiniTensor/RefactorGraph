#ifndef KERNEL_CUDA_FUNCTIONS_CUH
#define KERNEL_CUDA_FUNCTIONS_CUH

namespace refactor::kernel::cuda {

    int currentDevice();

    void sync();

    void setCudaDevice(int);

    void copyOut(void *dst, const void *src, size_t size);

    template<typename T, typename Tb>
    inline T round_up(T m, Tb d) {
        return ((m + T(d) - 1) / T(d)) * T(d);
    }
}// namespace refactor::kernel::cuda

#endif// KERNEL_CUDA_FUNCTIONS_CUH
