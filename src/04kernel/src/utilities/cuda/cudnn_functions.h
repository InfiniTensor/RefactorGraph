#ifndef KERNEL_CUDNN_FUNCTIONS_H
#define KERNEL_CUDNN_FUNCTIONS_H

#include "common/data_type.h"
#include "common/error_handler.h"
#include <cudnn.h>

#define CUDNN_ASSERT(STATUS) ASSERT((STATUS) == CUDNN_STATUS_SUCCESS, "cudnn not success")

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(common::DataType);

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_FUNCTIONS_H
