#ifndef KERNEL_TRANSPOSE_CUDA_KERNEL_HH
#define KERNEL_TRANSPOSE_CUDA_KERNEL_HH

#include "kernel/collectors/transpose.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct TransposeCuda final : public Kernel {
        DataType dataType;
        TransposeInfo info;

        TransposeCuda(DataType, TransposeInfo) noexcept;

        static KernelBox build(DataType, TransposeInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CUDA_KERNEL_HH
