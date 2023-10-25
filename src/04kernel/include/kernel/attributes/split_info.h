#ifndef KERNEL_SPLIT_INFO_H
#define KERNEL_SPLIT_INFO_H

#include "../collector.h"

namespace refactor::kernel {

    struct SplitInfo {
        uint_lv2 blockCount, sum;
        absl::InlinedVector<uint_lv2, 4> segments;

        SplitInfo(uint_lv2 axis, TensorRefs const &outputs) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
