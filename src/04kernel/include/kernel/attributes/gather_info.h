#ifndef KERNEL_GATHER_INFO_H
#define KERNEL_GATHER_INFO_H

#include "../tensor.h"

namespace refactor::kernel {

    struct GatherInfo {
        uint_lv2 prefix, postfix, midSize;
        absl::InlinedVector<uint_lv2, 4> strides;
        DataType idxType;

        GatherInfo(uint_lv2 axis, Tensor const &data, Tensor const &indices) noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_GATHER_INFO_H
