#include "functions.hh"

namespace refactor::hardware {

    int getDeviceCount() {
        int deviceCount;
        BANG_ASSERT(cnrtGetDeviceCount(&deviceCount));
        return deviceCount;
    }
    void setDevice(int device) {
        BANG_ASSERT(cnrtSetDevice(device));
    }
    MemInfo getMemInfo() {
        MemInfo memInfo;
        BANG_ASSERT(cudaMemGetInfo(&memInfo.free, &memInfo.total));
        return memInfo;
    }

}// namespace refactor::hardware
