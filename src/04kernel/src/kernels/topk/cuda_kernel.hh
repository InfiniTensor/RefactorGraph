#ifndef KERNEL_TOPK_CUDA_KERNEL_HH
#define KERNEL_TOPK_CUDA_KERNEL_HH

#include "kernel/attributes/topk_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct TopKCuda final : public Kernel {
        TopKInfo info;

        explicit TopKCuda(TopKInfo) noexcept;

        static KernelBox build(TopKInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TOPK_CUDA_KERNEL_HH
