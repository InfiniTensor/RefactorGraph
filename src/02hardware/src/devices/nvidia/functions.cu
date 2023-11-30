#include "functions.cuh"

namespace refactor::hardware {

    void setDevice(int device) {
        CUDA_ASSERT(cudaSetDevice(device));
    }
    int getDeviceCount() {
        int deviceCount;
        CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));
        return deviceCount;
    }

}// namespace refactor::hardware
