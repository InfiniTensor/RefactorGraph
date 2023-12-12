#ifndef KERNEL_SCATTER_ND_CPU_KERNEL_HH
#define KERNEL_SCATTER_ND_CPU_KERNEL_HH

#include "kernel/attributes/scatter_nd_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ScatterNDCpu final : public Kernel {
        ScatterNDInfo info;

        explicit ScatterNDCpu(decltype(info));

        static KernelBox build(decltype(info)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SCATTER_ND_CPU_KERNEL_HH
