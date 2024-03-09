#ifndef KERNEL_TOPK_CPU_KERNEL_HH
#define KERNEL_TOPK_CPU_KERNEL_HH

#include "kernel/attributes/topk_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct TopKCpu final : public Kernel {
        TopKInfo info;
        explicit TopKCpu(TopKInfo info) noexcept;

        static KernelBox build(TopKInfo info) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CPU_KERNEL_HH
