#ifndef KERNEL_MOE_CPU_KERNEL_HH
#define KERNEL_MOE_CPU_KERNEL_HH

#include "kernel/attributes/moe_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct AssignPosCpu final : public Kernel {
        AssignPosInfo info;
        explicit AssignPosCpu(AssignPosInfo info) noexcept;

        static KernelBox build(AssignPosInfo info) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

    struct ReorderCpu final : public Kernel {
        ReorderInfo info;
        explicit ReorderCpu(ReorderInfo info) noexcept;

        static KernelBox build(ReorderInfo info) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CPU_KERNEL_HH
