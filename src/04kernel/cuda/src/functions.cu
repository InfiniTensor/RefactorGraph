#include "kernel/cuda/functions.cuh"
#include "macro.cuh"
#include <cstdio>

namespace refactor::kernel::cuda {

    int currentDevice() {
        int device;
        CUDA_ASSERT(cudaGetDevice(&device));
        return device;
    }

    void sync() {
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    void copyOut(void *dst, const void *src, size_t size) {
        sync();
        CUDA_ASSERT(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }

    void setCudaDevice(int id) {
        cudaSetDevice(id);
    }

}// namespace refactor::kernel::cuda
