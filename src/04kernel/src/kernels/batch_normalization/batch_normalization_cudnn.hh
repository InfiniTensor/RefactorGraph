#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_HH
#define KERNEL_BATCH_NORMALIZATION_CUDNN_HH

#include "common/data_type.h"
#include "kernel/collectors/batch_normalization.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    struct BatchNormalizationCudnn final : public Kernel {
        float epsilon;
        common::DataType dataType;
        Shape shape;
        uint32_t valueSize;

        BatchNormalizationCudnn(float, common::DataType, Shape, uint32_t) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Operation lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_HH
