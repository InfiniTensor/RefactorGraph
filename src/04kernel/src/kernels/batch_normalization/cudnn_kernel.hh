#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH
#define KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH

#include "cudnn_impl.h"

namespace refactor::kernel {

    /// @brief Use `cudnnBatchNormalizationForwardInference`.
    ///        It only supports 4D and 5D tensor.
    struct BatchNormalizationCudnn final : public Kernel {
        cudnn::Info info;

        BatchNormalizationCudnn(cudnn::Info) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Operation lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH
