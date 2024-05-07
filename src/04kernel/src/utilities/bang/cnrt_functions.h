#ifndef KERNEL_CNRT_FUNCTIONS_H
#define KERNEL_CNRT_FUNCTIONS_H

#include "common.h"

namespace refactor::kernel::bang {

    int currentDevice();

    void sync();

    void copyOut(void *dst, const void *src, size_t size);

}// namespace refactor::kernel::bang

#endif// KERNEL_CNRT_FUNCTIONS_H
