﻿#ifndef KERNEL_SPLIT_INFO_H
#define KERNEL_SPLIT_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    /// @brief 优化用于计算的 Split 描述。
    struct SplitInfo {
        /// @brief 要拷贝的次数和每次拷贝的大小，即所有片段的总大小。
        ///        NOTICE 要拷贝的次数最小值可以取到 1，表示把数据分成几个连续的块。
        dim_t blockCount, sum;
        /// @brief 要拷贝的每个片段的大小，已经考虑了每个数据的大小。
        absl::InlinedVector<dim_t, 4> segments;

        SplitInfo(dim_t axis, TensorRefs const &outputs);
        dim_t unit(dim_t maxBlockSize) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
