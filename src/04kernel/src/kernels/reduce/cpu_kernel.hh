#ifndef KERNEL_REDUCE_CPU_KERNEL_HH
#define KERNEL_REDUCE_CPU_KERNEL_HH

#include "kernel/collectors/reduce.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ReduceCpu final : public Kernel {
        DataType dataType;
        std::vector<int64_t> axes;
        ReduceType reduceType;
        Shape shape;

        ReduceCpu(decltype(axes), ReduceType, DataType, Shape) noexcept;

        static KernelBox build(decltype(axes), ReduceType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        Routine lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_REDUCE_CPU_KERNEL_HH
