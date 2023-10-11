#ifndef KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH
#define KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH

#include "common/data_type.h"
#include "kernel/collectors/simple_unary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SimpleUnary final : public Kernel {
        common::DataType dataType;
        SimpleUnaryType opType;
        size_t size;

        SimpleUnary(SimpleUnaryType, common::DataType, size_t) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Operation lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH
