#ifndef KERNEL_BATCH_NORMALIZATION_CNNL_KERNEL_HH
#define KERNEL_BATCH_NORMALIZATION_CNNL_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    /// @brief Use `cnnlBatchNormalizationForwardInference`.
    ///        It only supports 4D and 5D tensors.
    struct BatchNormalizationCnnl final : public Kernel {
        struct {
            float epsilon;
            DataType dtX, dtP;
            LayoutType layout;
            int dimAx[4];// dimA for x
        } info;

        explicit BatchNormalizationCnnl(decltype(info)) noexcept;

        static KernelBox build(float, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_CNNL_KERNEL_HH
