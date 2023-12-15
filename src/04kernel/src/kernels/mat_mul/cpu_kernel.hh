#ifndef KERNEL_MATMUL_CPU_KERNEL_HH
#define KERNEL_MATMUL_CPU_KERNEL_HH

#include "kernel/attributes/mat_mul_info.h"
#include "kernel/kernel.h"

namespace refactor::kernel {

    struct MatMulCPU final : public Kernel {
        MatMulInfo info;

        explicit MatMulCPU(decltype(info)) noexcept;

        static KernelBox build(decltype(info)) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;

        RoutineWorkspace lower(Resources &) const noexcept final;
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_CPU_KERNEL_HH
