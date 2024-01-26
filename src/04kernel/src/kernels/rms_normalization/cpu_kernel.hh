#ifndef KERNEL_RMS_NORMALIZATION_CPU_KERNEL_HH
#define KERNEL_RMS_NORMALIZATION_CPU_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct RmsNormalizationCpu final : public Kernel {
        float epsilon;
        DataType dataType;
        dim_t blockCount, blockSize;

        RmsNormalizationCpu(
            decltype(epsilon),
            decltype(dataType),
            decltype(blockCount),
            decltype(blockSize)) noexcept;

        static KernelBox build(float, Tensor const &x) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_RMS_NORMALIZATION_CPU_KERNEL_HH
