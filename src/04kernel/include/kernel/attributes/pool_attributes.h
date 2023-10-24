#ifndef KERNEL_POOL_ATTRIBUTES_H
#define KERNEL_POOL_ATTRIBUTES_H

#include "common.h"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    enum class PoolType {
        Average,
        Lp,
        Max,
    };

    using KernelShape = absl::InlinedVector<uint_lv1, 2>;

    /// @brief 池化参数用于池化和卷积。
    class PoolAttributes {
        /// @brief 池化参数包含 `(1+1+2)rank` 个元素。
        ///        由于 rank 常常取 4，参数总数也往往至少有 16 个。
        ///        如果使用 uint16_t 并 inline，则共 16x2+8 = 40 字节，
        ///        这样拷贝开销还是可以接受的。
        absl::InlinedVector<uint_lv1, 16> _values;

    public:
        PoolAttributes(
            size_t rank,
            int64_t const *dilations,
            int64_t const *pads,
            int64_t const *strides);

        size_t rank() const noexcept;
        uint_lv1 const *dilations() const noexcept;
        uint_lv1 const *pads() const noexcept;
        uint_lv1 const *padsBegin() const noexcept;
        uint_lv1 const *padsEnd() const noexcept;
        uint_lv1 const *strides() const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_ATTRIBUTES_H
