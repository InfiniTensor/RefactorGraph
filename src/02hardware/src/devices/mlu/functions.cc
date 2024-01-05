#include "functions.hh"

namespace refactor::hardware {

    int getDeviceCount() {
        unsigned deviceCount;
        BANG_ASSERT(cnrtGetDeviceCount(&deviceCount));
        return static_cast<int>(deviceCount);
    }
    void setDevice(int device) {
        BANG_ASSERT(cnrtSetDevice(device));
    }
    MemInfo getMemInfo() {
        MemInfo memInfo;
        BANG_ASSERT(cnrtMemGetInfo(&memInfo.free, &memInfo.total));
        return memInfo;
    }

}// namespace refactor::hardware
