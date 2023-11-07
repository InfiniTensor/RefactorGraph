#ifndef KERNEL_SOFTMAX_INFO_H
#define KERNEL_SOFTMAX_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct SoftmaxInfo {
        uint_lv2 pre, mid, post, size;
        DataType type;

        SoftmaxInfo(Tensor const &data, uint_lv2 axis) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_INFO_H
