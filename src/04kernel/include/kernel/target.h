#ifndef KRENEL_TARGET_H
#define KRENEL_TARGET_H

#include "mem_manager/mem_functions.h"
#include <cstdint>

namespace refactor::kernel {
    using mem_manager::MemFunctions;

    struct Target {
        enum : uint8_t {
            Cpu,
            NvidiaGpu,
        } internal;

        constexpr Target(decltype(internal) i) noexcept
            : internal(i) {}

        constexpr operator decltype(internal)() const noexcept {
            return internal;
        }

        MemFunctions memFunc() const;
    };

}// namespace refactor::kernel

#endif// KRENEL_TARGET_H
