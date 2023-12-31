#ifndef KERNEL_REDUCE_CPU_KERNEL_HH
#define KERNEL_REDUCE_CPU_KERNEL_HH

#include "kernel/collectors/reduce.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ReduceCpu final : public Kernel {
        DataType dataType;
        ReduceType reduceType;
        Axes axes;
        Shape shape;

        ReduceCpu(decltype(dataType),
                  decltype(reduceType),
                  decltype(axes),
                  decltype(shape)) noexcept;

        static KernelBox build(decltype(axes), ReduceType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_REDUCE_CPU_KERNEL_HH
