#ifndef KERNEL_ACTIVATION_CUDNN_KERNEL_HH
#define KERNEL_ACTIVATION_CUDNN_KERNEL_HH

#include "kernel/collectors/simple_unary.h"

namespace refactor::kernel {

    struct ActivationCudnn final : public Kernel {
        SimpleUnaryType type;
        DataType dataType;
        int size;

        ActivationCudnn(SimpleUnaryType, DataType, int) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_ACTIVATION_CUDNN_KERNEL_HH
