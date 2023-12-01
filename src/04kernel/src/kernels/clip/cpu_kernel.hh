#ifndef KERNEL_CLIP_CPU_KERNEL_HH
#define KERNEL_CLIP_CPU_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ClipCpu final : public Kernel {
        DataType dataType;
        size_t size;
        bool hasMax;

        ClipCpu(decltype(dataType), decltype(size), decltype(hasMax)) noexcept;

        static KernelBox build(Tensor const &, bool hasMax) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_CPU_KERNEL_HH
