#ifndef KERNEL_WHERE_INFO_H
#define KERNEL_WHERE_INFO_H

#include "common.h"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    using Shape = absl::InlinedVector<uint_lv2, 4>;
    struct WhereBroadcast {
        std::vector<uint_lv2> _strides;
        uint_lv2 _size;
        struct Triplet {
            uint_lv2 c_index;
            uint_lv2 x_index;
            uint_lv2 y_index;
        };

        WhereBroadcast(Shape const &, Shape const &, Shape const &, Shape const &) noexcept;
        Triplet locate(uint_lv2) const noexcept;
        uint_lv2 size() const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_WHERE_INFO_H
