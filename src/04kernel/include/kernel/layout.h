#ifndef KERNEL_LAYOUT_H
#define KERNEL_LAYOUT_H

#include "common.h"
#include <array>
#include <cstdint>

namespace refactor::kernel {

    struct LayoutType {
        enum : uint8_t {
            NCHW,
            NHWC,
            Others,
        } internal;

        constexpr LayoutType(decltype(internal) i) noexcept : internal(i) {}
        constexpr operator decltype(internal)() const noexcept { return internal; }

        constexpr static auto permutation(
            decltype(internal) lhs,
            decltype(internal) rhs) -> std::array<uint8_t, 4> {
#define MERGE(FROM, TO) (static_cast<int>(FROM) << 8) | static_cast<int>(TO)
            switch (MERGE(lhs, rhs)) {
                case MERGE(NCHW, NHWC):
                    return {0, 3, 1, 2};
                case MERGE(NHWC, NCHW):
                    return {0, 2, 3, 1};
                case MERGE(NCHW, NCHW):
                case MERGE(NHWC, NHWC):
                    return {0, 1, 2, 3};
                default:
                    UNREACHABLE();
            }
#undef MERGE
        }
    };

#define PERMUTATION(FROM, TO) LayoutType::permutation(LayoutType::FROM, LayoutType::TO)

}// namespace refactor::kernel

#endif// KERNEL_LAYOUT_H
