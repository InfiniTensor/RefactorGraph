#include "kernel/cuda/functions.cuh"

namespace refactor::kernel::cuda {

    void sync() {
        cudaDeviceSynchronize();
    }

}// namespace refactor::kernel::cuda
