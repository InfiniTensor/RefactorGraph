#ifndef HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH
#define HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH

#include "common.h"

#define CUDA_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != cudaSuccess) {                             \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cudaGetErrorString(status), (int) status));        \
    }

namespace refactor::hardware {

    void setDevice(int device);
    int getDeviceCount();

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH
