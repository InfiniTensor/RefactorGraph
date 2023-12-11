#ifndef KERNEL_CUDNN_FUNCTIONS_H
#define KERNEL_CUDNN_FUNCTIONS_H

#include "common.h"
#include <cudnn.h>

#define CUDNN_ASSERT(STATUS)                                      \
    if (auto status = (STATUS); status != CUDNN_STATUS_SUCCESS) { \
        fmt::println("cudnn failed on \"" #STATUS "\" with {}",   \
                     cudnnGetErrorString(status));                \
        abort();                                                  \
    }

namespace refactor::kernel::cudnn {

    cudnnDataType_t cudnnDataTypeConvert(DataType);

    // A helper function that set CuDNN tensor descriptor given tensor shape and type
    void setCudnnTensor(cudnnTensorDescriptor_t, DataType, slice_t<int>);

    template<class T>
    constexpr uint64_t factor(T x) noexcept {
        static_assert(std::is_floating_point_v<T>);
        static_assert(sizeof(T) <= sizeof(uint64_t));
        union {
            T f;
            uint64_t i;
        } u{x};
        return u.i;
    }

}// namespace refactor::kernel::cudnn

#endif// KERNEL_CUDNN_FUNCTIONS_H
