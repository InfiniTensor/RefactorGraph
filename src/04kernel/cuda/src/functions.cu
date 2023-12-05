#include "kernel/cuda/functions.cuh"
#include <cstdio>

namespace refactor::kernel::cuda {

    void sync() {
        auto state = cudaDeviceSynchronize();
        if (state != cudaSuccess) {
            printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(state));
            exit(1);
        }
    }

    void copyOut(void *dst, const void *src, size_t size) {
        sync();
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }

    void setCudaDevice(int id) {
        cudaSetDevice(id);
    }

}// namespace refactor::kernel::cuda
