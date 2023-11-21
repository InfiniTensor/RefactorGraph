#ifndef KERNEL_EXPAND_CUDA_KERNEL_HH
#define KERNEL_EXPAND_CUDA_KERNEL_HH

#include "kernel/attributes/expand_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct ExpandCuda final : public Kernel {
        ExpandInfo info;

        explicit ExpandCuda(ExpandInfo) noexcept;

        static KernelBox build(ExpandInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_EXPAND_CUDA_KERNEL_HH
