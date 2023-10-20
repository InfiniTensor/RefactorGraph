#ifndef KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH
#define KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH

#include "kernel/collectors/simple_unary.h"
#include "kernel/tensor.h"
#include "common.h"

namespace refactor::kernel {

    struct SimpleUnaryCpu final : public Kernel {
        DataType dataType;
        SimpleUnaryType opType;
        size_t size;

        SimpleUnaryCpu(SimpleUnaryType, DataType, size_t) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_CPU_KERNEL_HH
