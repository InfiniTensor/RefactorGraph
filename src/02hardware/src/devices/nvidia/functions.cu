#include "functions.cuh"

namespace refactor::hardware {

    void setDevice(int device) {
        CUDA_ASSERT(cudaSetDevice(device));
    }

}// namespace refactor::hardware
