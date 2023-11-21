#ifndef KERNEL_SIMPLE_UNARY_CUDA_KERNEL_HH
#define KERNEL_SIMPLE_UNARY_CUDA_KERNEL_HH

#include "kernel/collectors/simple_unary.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct SimpleUnaryCuda final : public Kernel {
        DataType dataType;
        SimpleUnaryType opType;
        size_t size;

        SimpleUnaryCuda(SimpleUnaryType, DataType, size_t) noexcept;

        static KernelBox build(SimpleUnaryType, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SIMPLE_UNARY_CUDA_KERNEL_HH
