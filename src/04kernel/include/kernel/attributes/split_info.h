#ifndef KERNEL_SPLIT_INFO_H
#define KERNEL_SPLIT_INFO_H

#include "../collector.h"

namespace refactor::kernel {

    class SplitInfo {
        std::vector<uint_lv2> _values;
        uint_lv2 _prefixLen;

    public:
        SplitInfo(uint_lv2 axis, TensorRefs const &outputs) noexcept;
        slice_t<uint_lv2> prefix() const noexcept;
        slice_t<uint_lv2> postfix() const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_INFO_H
