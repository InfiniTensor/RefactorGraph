#ifndef KERNEL_SIMPLE_UNARY_CNNL_KERNEL_HH
#define KERNEL_SIMPLE_UNARY_CNNL_KERNEL_HH

#include "kernel/collectors/simple_unary.h"

namespace refactor::kernel {

    struct SimpleUnaryCnnl final : public Kernel {
        SimpleUnaryType type;
        DataType dataType;
        int size;

        SimpleUnaryCnnl(SimpleUnaryType, DataType, int) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_CNNL_KERNEL_HH
