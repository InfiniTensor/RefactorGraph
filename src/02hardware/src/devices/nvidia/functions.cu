#include "functions.cuh"

namespace refactor::hardware {

    int getDeviceCount() {
        int deviceCount;
        CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));
        return deviceCount;
    }
    void setDevice(int device) {
        CUDA_ASSERT(cudaSetDevice(device));
    }
    MemInfo getMemInfo() {
        MemInfo memInfo;
        CUDA_ASSERT(cudaMemGetInfo(&memInfo.free, &memInfo.total));
        return memInfo;
    }

}// namespace refactor::hardware
