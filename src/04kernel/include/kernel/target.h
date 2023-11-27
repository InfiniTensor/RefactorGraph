#ifndef KRENEL_TARGET_H
#define KRENEL_TARGET_H

#include "hardware/device.h"

namespace refactor::kernel {

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

        Arc<hardware::Device> device() const;
    };

}// namespace refactor::kernel

#endif// KRENEL_TARGET_H
