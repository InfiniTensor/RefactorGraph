#ifndef KERNEL_PAD_CPU_KERNEL_HH
#define KERNEL_PAD_CPU_KERNEL_HH

#include "kernel/attributes/pad_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct PadCpu final : public Kernel {
        PadInfo info;
        PadType mode;
        size_t valueLength;

        explicit PadCpu(PadInfo, PadType, size_t) noexcept;

        static KernelBox build(PadInfo, PadType, std::optional<std::reference_wrapper<Tensor const>>) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_PAD_CPU_KERNEL_HH

