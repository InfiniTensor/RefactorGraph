#ifndef KERNEL_SPLIT_CPU_KERNEL_HH
#define KERNEL_SPLIT_CPU_KERNEL_HH

#include "kernel/attributes/slice_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct SliceCpu final : public Kernel {
        SliceInfo info;

        explicit SliceCpu(SliceInfo) noexcept;

        static KernelBox build(SliceInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CPU_KERNEL_HH
