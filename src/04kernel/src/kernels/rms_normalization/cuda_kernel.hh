#ifndef KERNEL_RMS_NORMALIZATION_CUDA_KERNEL_HH
#define KERNEL_RMS_NORMALIZATION_CUDA_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    struct RmsNormalizationCuda final : public Kernel {
        float epsilon;
        DataType dataType;
        dim_t blockCount, blockSize;

        RmsNormalizationCuda(
            decltype(epsilon),
            decltype(dataType),
            decltype(blockCount),
            decltype(blockSize)) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_RMS_NORMALIZATION_CUDA_KERNEL_HH
