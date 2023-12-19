#ifndef KERNEL_DEQUANTIZE_LINEAR_CUDA_KERNEL_HH
#define KERNEL_DEQUANTIZE_LINEAR_CUDA_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct DequantizeLinearCuda final : public Kernel {
        DataType from, to;
        size_t size;
        bool withZeroPoint;

        DequantizeLinearCuda(
            decltype(from),
            decltype(to),
            decltype(size),
            decltype(withZeroPoint)) noexcept;

        static KernelBox build(TensorRefs const &, Tensor const &) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_DEQUANTIZE_LINEAR_CUDA_KERNEL_HH
