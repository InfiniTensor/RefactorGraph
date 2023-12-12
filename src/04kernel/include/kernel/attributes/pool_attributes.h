#ifndef KERNEL_POOL_ATTRIBUTES_H
#define KERNEL_POOL_ATTRIBUTES_H

#include "common.h"

namespace refactor::kernel {

    struct PoolType {
        enum : uint8_t {
            Average,
            Lp,
            Max,
        } type;

        constexpr PoolType() noexcept
            : type(Average) {}
        constexpr PoolType(decltype(type) type_) noexcept
            : type(type_) {}
        constexpr operator decltype(type)() const noexcept {
            return type;
        }
        constexpr std::string_view toString() const noexcept {
            switch (type) {
                case Average:
                    return "Average";
                case Lp:
                    return "Lp";
                case Max:
                    return "Max";
                default:
                    UNREACHABLE();
            }
        }
    };

    using KernelShape = absl::InlinedVector<ddim_t, 2>;

    /// @brief 池化参数用于池化和卷积。
    class PoolAttributes {
        /// @brief 池化参数包含 `(1+1+2)rank` 个元素。
        ///        由于 rank 常常取 4，参数总数也往往至少有 16 个。
        ///        如果使用 ddim_t 并 inline，则共 16x2+8 = 40 字节，
        ///        这样拷贝开销还是可以接受的。
        absl::InlinedVector<ddim_t, 16> _values;

    public:
        PoolAttributes(
            size_t rank,
            int64_t const *dilations,
            int64_t const *pads,
            int64_t const *strides);

        size_t rank() const noexcept;
        ddim_t const *dilations() const noexcept;
        ddim_t const *pads() const noexcept;
        ddim_t const *padsBegin() const noexcept;
        ddim_t const *padsEnd() const noexcept;
        ddim_t const *strides() const noexcept;
        std::string toString() const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_POOL_ATTRIBUTES_H
