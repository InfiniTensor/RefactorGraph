#ifndef KERNEL_MATMUL_CPU_KERNEL_HH
#define KERNEL_MATMUL_CPU_KERNEL_HH

#include "kernel/attributes/matmul_info.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct MatMulCPU final : public Kernel {
        MatMulInfo info;

        explicit MatMulCPU(MatMulInfo) noexcept;

        static KernelBox build(Tensor const &, Tensor const &, Tensor const &, MatMulInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        Routine lower() const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_CPU_KERNEL_HH
