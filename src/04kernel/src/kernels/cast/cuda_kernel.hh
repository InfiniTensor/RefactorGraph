#ifndef KERNEL_CLIP_CUDA_KERNEL_HH
#define KERNEL_CLIP_CUDA_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct CastCuda final : public Kernel {
        DataType from, to;
        size_t size;

        CastCuda(decltype(from), decltype(to), decltype(size)) noexcept;

        static KernelBox build(Tensor const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CLIP_CUDA_KERNEL_HH
