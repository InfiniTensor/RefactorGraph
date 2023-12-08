#ifndef KERNEL_CLIP_CPU_KERNEL_HH
#define KERNEL_CLIP_CPU_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct CastCpu final : public Kernel {
        DataType from, to;
        size_t size;

        CastCpu(decltype(from), decltype(to), decltype(size)) noexcept;

        static KernelBox build(Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_CPU_KERNEL_HH
