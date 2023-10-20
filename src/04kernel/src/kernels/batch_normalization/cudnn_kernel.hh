#ifndef KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH
#define KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include "common.h"

namespace refactor::kernel {
    /// @brief Use `cudnnBatchNormalizationForwardInference`.
    ///        It only supports 4D and 5D tensors.
    struct BatchNormalizationCudnn final : public Kernel {
        struct {
            float epsilon;
            DataType dtX, dtParam;
            LayoutType layout;
            int dimAx[4];// dimA for x, cudnn naming convension
        } info;

        explicit BatchNormalizationCudnn(decltype(info)) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CUDNN_KERNEL_HH
