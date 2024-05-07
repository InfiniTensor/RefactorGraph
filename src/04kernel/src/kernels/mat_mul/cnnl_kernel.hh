#ifndef KERNEL_MATMUL_CNNL_KERNEL_HH
#define KERNEL_MATMUL_CNNL_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct MatMulCnnl final : public Kernel {
        struct {
            DataType dataType;
            bool transA, transB;
            float alpha, beta;
            std::vector<int> aDim, bDim, cDim;
            std::optional<std::vector<int>> biasDim;
        } info;

        explicit MatMulCnnl(decltype(info)) noexcept;

        static KernelBox build(TensorRefs, TensorRefs, bool, bool, float, float) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_CNNL_KERNEL_HH
