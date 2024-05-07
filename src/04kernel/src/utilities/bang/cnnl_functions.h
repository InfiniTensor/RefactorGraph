#ifndef KERNEL_CNNL_FUNCTIONS_H
#define KERNEL_CNNL_FUNCTIONS_H

#include "common.h"
#include <cnnl.h>

#define BANG_ASSERT(STATUS)                                                          \
    if (auto status = (STATUS); status != CNRT_RET_SUCCESS) {                        \
        RUNTIME_ERROR(fmt::format("bang failed on \"" #STATUS "\" with \"{}\" ({})", \
                                  cnrtGetErrorStr(status), (int) status));           \
    }

#define CNNL_ASSERT(STATUS)                                      \
    if (auto status = (STATUS); status != CNNL_STATUS_SUCCESS) { \
        fmt::println("cnnl failed on \"" #STATUS "\" with {}",  \
                     cnnlGetErrorString(status));                \
        abort();                                                 \
    }

namespace refactor::kernel::cnnl {

    cnnlDataType_t cnnlDataTypeConvert(DataType);

    // A helper function that set Cnnl tensor descriptor given tensor shape and type
    void setCnnlTensor(cnnlTensorDescriptor_t, DataType, slice_t<int>);

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

}// namespace refactor::kernel::cnnl

#endif// KERNEL_CNNL_FUNCTIONS_H
