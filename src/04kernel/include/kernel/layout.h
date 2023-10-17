#ifndef KERNEL_LAYOUT_H
#define KERNEL_LAYOUT_H

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

        std::array<uint8_t, 4> permTo(LayoutType rhs) const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_LAYOUT_H
