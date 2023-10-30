#ifndef KERNEL_GATHER_CPU_KERNEL_HH
#define KERNEL_GATHER_CPU_KERNEL_HH

#include "kernel/attributes/gather_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct GatherCpu final : public Kernel {
        GatherInfo info;

        explicit GatherCpu(GatherInfo) noexcept;

        static KernelBox build(GatherInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_GATHER_CPU_KERNEL_HH
