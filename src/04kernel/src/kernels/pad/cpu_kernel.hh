#ifndef KERNEL_PAD_CPU_KERNEL_HH
#define KERNEL_PAD_CPU_KERNEL_HH

#include "kernel/attributes/pad_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct PadCpu final : public Kernel {
        PadInfo info;

        explicit PadCpu(PadInfo) noexcept;

        static KernelBox build(PadInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_PAD_CPU_KERNEL_HH