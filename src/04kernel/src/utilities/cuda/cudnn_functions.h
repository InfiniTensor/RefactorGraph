#ifndef KERNEL_CUDNN_FUNCTIONS_H
#define KERNEL_CUDNN_FUNCTIONS_H

#include "refactor/common.h"
#include "refactor/common.h"
#include <cudnn.h>

#define CUDNN_ASSERT(STATUS) ASSERT((STATUS) == CUDNN_STATUS_SUCCESS, "cudnn not success")

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(DataType);

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_FUNCTIONS_H
