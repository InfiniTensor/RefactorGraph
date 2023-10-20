#ifndef KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH
#define KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include "common.h"

namespace refactor::kernel {

    struct BatchNormalization final : public Kernel {
        float epsilon;
        DataType dts[3];
        Shape shape;

        BatchNormalization(float, DataType, DataType, DataType, Shape) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH
