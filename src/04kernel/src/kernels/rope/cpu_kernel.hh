#ifndef KERNEL_ROPE_CPU_KERNEL_HH
#define KERNEL_ROPE_CPU_KERNEL_HH


#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include "kernel/attributes/rope_info.h"

namespace refactor::kernel {
    struct RoPECpu final : public Kernel {
        RoPEInfo info;
        RoPECpu(decltype(info)) noexcept;

        static KernelBox build(decltype(info), Tensor const &x) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const final;

    };
}

#endif
