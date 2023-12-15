#ifndef KERNEL_MAT_MUL_INTEGER_INFO_H
#define KERNEL_MAT_MUL_INTEGER_INFO_H

#include "kernel/attributes/broadcaster.h"

namespace refactor::kernel {

    struct MatMulIntegerInfo {
        struct Input {
            bool signed_;
            bool withZeroPoint;

            Input(TensorRefs const &, size_t i) noexcept;
        };

        Input a, b;
        dim_t m, k, n;
        Broadcaster broadcaster;

        explicit MatMulIntegerInfo(TensorRefs const &inputs) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_MAT_MUL_INTEGER_INFO_H
