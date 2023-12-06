#ifndef KERNEL_CLIP_CUDA_KERNEL_HH
#define KERNEL_CLIP_CUDA_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ClipCuda final : public Kernel {
        DataType dataType;
        size_t size;
        bool hasMax;

        ClipCuda(decltype(dataType), decltype(size), decltype(hasMax)) noexcept;

        static KernelBox build(Tensor const &, bool hasMax) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_CUDA_KERNEL_HH
