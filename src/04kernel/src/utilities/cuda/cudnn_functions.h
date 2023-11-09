#ifndef KERNEL_CUDNN_FUNCTIONS_H
#define KERNEL_CUDNN_FUNCTIONS_H

#include "common.h"
#include <cudnn.h>

#define CUDNN_ASSERT(STATUS)                                                 \
    if (auto status = (STATUS); status != CUDNN_STATUS_SUCCESS) {            \
        RUNTIME_ERROR(fmt::format("cudnn failed on \"" #STATUS "\" with {}", \
                                  cudnnGetErrorString(status)));             \
    }

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(DataType);

    // A helper function that set CuDNN tensor descriptor given tensor shape and type
    void setCudnnTensor(cudnnTensorDescriptor_t aDesc, DataType dt, int const *dims, size_t rank);

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_FUNCTIONS_H
