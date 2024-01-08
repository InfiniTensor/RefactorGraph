#include "functions.hh"

namespace refactor::hardware {

#ifdef USE_BANG
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
#endif

}// namespace refactor::hardware
