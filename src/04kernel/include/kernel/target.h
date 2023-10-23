#ifndef KRENEL_TARGET_H
#define KRENEL_TARGET_H

#include "common.h"
#include "mem_manager/mem_manager.hh"

namespace refactor::kernel {
    using mem_manager::MemManager;

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

        Arc<MemManager> memManager() const;
    };

}// namespace refactor::kernel

#endif// KRENEL_TARGET_H
