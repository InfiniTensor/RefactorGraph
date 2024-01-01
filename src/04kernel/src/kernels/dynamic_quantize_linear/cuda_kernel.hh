#ifndef KERNEL_DYNAMIC_QUANTIZE_LINEAR_CUDA_KERNEL_HH
#define KERNEL_DYNAMIC_QUANTIZE_LINEAR_CUDA_KERNEL_HH

#include "kernel/kernel.h"

namespace refactor::kernel {

    struct DynamicQuantizeLinearCuda final : public Kernel {
        size_t size;

        explicit DynamicQuantizeLinearCuda(decltype(size)) noexcept;

        static KernelBox build(decltype(size)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_DYNAMIC_QUANTIZE_LINEAR_CUDA_KERNEL_HH
