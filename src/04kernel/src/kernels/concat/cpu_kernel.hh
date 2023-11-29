#ifndef KERNEL_CONCAT_CPU_KERNEL_HH
#define KERNEL_CONCAT_CPU_KERNEL_HH

#include "kernel/attributes/split_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ConcatCpu final : public Kernel {
        SplitInfo info;

        explicit ConcatCpu(SplitInfo) noexcept;

        static KernelBox build(SplitInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONCAT_CPU_KERNEL_HH
