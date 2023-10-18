#ifndef KERNEL_CUDNN_FUNCTIONS_H
#define KERNEL_CUDNN_FUNCTIONS_H

#include "refactor/common.h"
#include <cudnn.h>

#define CUDNN_ASSERT(STATUS)                                                 \
    if (auto status = (STATUS); status != CUDNN_STATUS_SUCCESS) {            \
        RUNTIME_ERROR(fmt::format("cudnn failed on \"" #STATUS "\" with {}", \
                                  cudnnGetErrorString(status)));             \
    }

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(DataType);

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_FUNCTIONS_H
