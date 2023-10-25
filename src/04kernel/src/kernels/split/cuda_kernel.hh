#ifndef KERNEL_SPLIT_CUDA_KERNEL_HH
#define KERNEL_SPLIT_CUDA_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SplitCuda final : public Kernel {
        DataType dataType;

        SplitCuda(DataType) noexcept;

        static KernelBox build(DataType) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_CUDA_KERNEL_HH
