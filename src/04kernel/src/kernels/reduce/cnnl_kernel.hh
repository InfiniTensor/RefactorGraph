#ifndef KERNEL_REDUCE_MEAN_CNNL_KERNEL_HH
#define KERNEL_REDUCE_MEAN_CNNL_KERNEL_HH

#include "kernel/collectors/reduce.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ReduceCnnl final : public Kernel {
        DataType dataType;
        ReduceType reduceType;
        Axes axes;
        Shape shape;

        ReduceCnnl(decltype(dataType),
                    decltype(reduceType),
                    decltype(axes),
                    decltype(shape)) noexcept;

        static KernelBox build(decltype(axes), ReduceType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };
}// namespace refactor::kernel

#endif// KERNEL_REDUCE_MEAN_CNNL_KERNEL_HH
