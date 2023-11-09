#ifndef KERNEL_REDUCE_MEAN_CUDNN_KERNEL_HH
#define KERNEL_REDUCE_MEAN_CUDNN_KERNEL_HH

#include "kernel/collectors/reduce.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct ReduceCudnn final : public Kernel {
        DataType dataType;
        Axes axes;
        ReduceType reduceType;
        Shape shape;

        ReduceCudnn(decltype(axes), ReduceType, DataType, Shape) noexcept;

        static KernelBox build(decltype(axes), ReduceType, TensorRefs) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &) const noexcept final;
#endif
    };
}// namespace refactor::kernel

#endif// KERNEL_REDUCE_MEAN_CUDNN_KERNEL_HH
