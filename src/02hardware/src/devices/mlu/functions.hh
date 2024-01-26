#ifndef HARDWARE_DEVICES_MLU_FUNCTIONS_CUH
#define HARDWARE_DEVICES_MLU_FUNCTIONS_CUH

#include "common.h"

#ifdef USE_BANG
#include "cnrt.h"

#define BANG_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != CNRT_RET_SUCCESS) {                        \
        RUNTIME_ERROR(fmt::format("bang failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cnrtGetErrorStr(status), (int) status));           \
    }
#endif

namespace refactor::hardware {

    struct MemInfo {
        size_t free, total;
    };

    int getDeviceCount();
    void setDevice(int device);
    MemInfo getMemInfo();

}// namespace refactor::hardware

#endif// HARDWARE_DEVICES_NVIDIA_FUNCTIONS_CUH
