#ifndef KERNEL_EXPAND_CPU_KERNEL_HH
#define KERNEL_EXPAND_CPU_KERNEL_HH

#include "kernel/attributes/expand_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ExpandCpu final : public Kernel {
        ExpandInfo info;

        explicit ExpandCpu(ExpandInfo) noexcept;

        static KernelBox build(ExpandInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_EXPAND_CPU_KERNEL_HH
