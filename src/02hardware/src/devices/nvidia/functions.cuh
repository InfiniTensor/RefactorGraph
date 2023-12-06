#ifndef HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH
#define HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH

#include "common.h"

#define CUDA_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != cudaSuccess) {                             \
        RUNTIME_ERROR(fmt::format("cuda failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cudaGetErrorString(status), (int) status));        \
    }

namespace refactor::hardware {

    struct MemInfo {
        size_t free, total;
    };

    int getDeviceCount();
    void setDevice(int device);
    MemInfo getMemInfo();

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH
