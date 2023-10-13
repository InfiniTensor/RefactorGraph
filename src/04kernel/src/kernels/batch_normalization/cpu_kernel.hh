#ifndef KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH
#define KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH

#include "common/data_type.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct BatchNormalization final : public Kernel {
        float epsilon;
        std::array<common::DataType, 3> dts;
        Shape shape;
        uint32_t paramSize;

        BatchNormalization(float, std::array<common::DataType, 3>, Shape, uint32_t) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CPU_KERNEL_HH
