#ifndef KERNEL_DEQUANTIZE_LINEAR_CPU_KERNEL_HH
#define KERNEL_DEQUANTIZE_LINEAR_CPU_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct DequantizeLinearCpu final : public Kernel {
        DataType from;
        size_t size;
        bool withZeroPoint;

        DequantizeLinearCpu(
            decltype(from),
            decltype(size),
            decltype(withZeroPoint)) noexcept;

        static KernelBox build(TensorRefs const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_DEQUANTIZE_LINEAR_CPU_KERNEL_HH
