#ifndef KERNEL_SOFTMAX_INFO_H
#define KERNEL_SOFTMAX_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct SoftmaxInfo {
        dim_t pre, mid, post;
        DataType type;

        SoftmaxInfo(Tensor const &data, dim_t axis) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_INFO_H
