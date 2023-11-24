#ifndef KERNEL_SPLIT_CUDA_KERNEL_HH
#define KERNEL_SPLIT_CUDA_KERNEL_HH

#include "kernel/attributes/split_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct SplitCuda final : public Kernel {
        SplitInfo info;

        explicit SplitCuda(SplitInfo) noexcept;

        static KernelBox build(SplitInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CUDA_KERNEL_HH
