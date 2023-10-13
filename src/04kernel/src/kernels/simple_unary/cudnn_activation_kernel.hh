#ifndef KERNEL_CONV_CUDNN_KERNEL_HH
#define KERNEL_CONV_CUDNN_KERNEL_HH

#include "kernel/collectors/simple_unary.h"

namespace refactor::kernel {

    struct ActivationCudnn final : public Kernel {
        SimpleUnaryType type;
        common::DataType dataType;
        size_t size;

        ActivationCudnn(SimpleUnaryType, common::DataType, size_t) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CUDNN_KERNEL_HH
