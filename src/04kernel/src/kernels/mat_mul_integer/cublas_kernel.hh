#ifndef KERNEL_MATMUL_CUBLAS_KERNEL_HH
#define KERNEL_MATMUL_CUBLAS_KERNEL_HH

#include "kernel/attributes/mat_mul_integer_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct MatMulIntegerCublas final : public Kernel {
        MatMulIntegerInfo info;

        explicit MatMulIntegerCublas(decltype(info)) noexcept;

        static KernelBox build(decltype(info)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_CUBLAS_KERNEL_HH
