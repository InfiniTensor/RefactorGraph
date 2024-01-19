#ifdef USE_BANG
#include "cnrt_functions.h"
#include "cnnl_functions.h"
#include <cnrt.h>
#include <cstdio>

namespace refactor::kernel::cnnl {

    int currentDevice() {
        int device;
        BANG_ASSERT(cnrtGetDevice(&device));
        return device;
    }

    void sync() {
        BANG_ASSERT(cnrtSyncDevice());
    }

    void copyOut(void *dst, const void *src, size_t size) {
        sync();
        BANG_ASSERT(cnrtMemcpy(dst, const_cast<void *>(src), size,
                               CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

}// namespace refactor::kernel::cnnl

#endif
