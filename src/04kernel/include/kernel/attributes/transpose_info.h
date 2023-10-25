#ifndef KERNEL_TRANPOSE_INFO_H
#define KERNEL_TRANPOSE_INFO_H

#include "common.h"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    using Shape = absl::InlinedVector<uint_lv2, 4>;
    using Permutation = Shape;

    /// @brief 转置参数优化。
    struct TransposeInfo {
        struct Dimension {
            uint_lv2 strideI, strideO;
        };

        /// @brief 转置信息包含 `(1+1)rank` 个元素。
        ///        由于 rank 常常取 4，参数总数也往往至少有 8 个。
        ///        如果使用 uint32_t 并 inline，则共 8x4+8 = 40 字节，
        ///        这样拷贝开销还是可以接受的。
        absl::InlinedVector<Dimension, 4> dims;
        uint_lv2 size;

        TransposeInfo(Shape const &, Permutation const &) noexcept;
        uint_lv2 locate(uint_lv2) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANPOSE_INFO_H
