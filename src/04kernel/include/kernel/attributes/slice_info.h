#ifndef KERNEL_SLICE_INFO_H
#define KERNEL_SLICE_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    /// @brief 优化用于计算的 Slice 描述。
    struct SliceInfo {
        struct Dim {
            int64_t start, step, length;
        };
    };

}// namespace refactor::kernel

#endif// KERNEL_SLICE_INFO_H
