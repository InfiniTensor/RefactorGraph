#ifndef KERNEL_TRANSPOSE_CUDA_KERNEL_HH
#define KERNEL_TRANSPOSE_CUDA_KERNEL_HH

#include "kernel/collectors/transpose.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct TransposeCuda final : public Kernel {
        struct
        {
        } info;

        explicit TransposeCuda(decltype(info)) noexcept;

        static KernelBox build(Permutation const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CUDA_KERNEL_HH
