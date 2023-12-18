#ifndef KERNEL_DYNAMIC_QUANTIZE_LINEAR_CPU_KERNEL_HH
#define KERNEL_DYNAMIC_QUANTIZE_LINEAR_CPU_KERNEL_HH

#include "kernel/kernel.h"

namespace refactor::kernel {

    struct DynamicQuantizeLinearCpu final : public Kernel {
        size_t size;

        explicit DynamicQuantizeLinearCpu(decltype(size)) noexcept;

        static KernelBox build(decltype(size)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_CPU_KERNEL_HH
