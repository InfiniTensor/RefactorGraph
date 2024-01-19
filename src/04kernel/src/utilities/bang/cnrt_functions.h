#ifndef KERNEL_CNRT_FUNCTIONS_H
#define KERNEL_CNRT_FUNCTIONS_H

#include "common.h"

namespace refactor::kernel::cnnl {

    int currentDevice();

    void sync();

    void copyOut(void *dst, const void *src, size_t size);

}// namespace refactor::kernel::cnnl

#endif// KERNEL_CNRT_FUNCTIONS_H
