#ifndef KERNEL_CONV_CUDNN_KERNEL_HH
#define KERNEL_CONV_CUDNN_KERNEL_HH

#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    struct MatMulCublas final : public Kernel {
        struct {
            DataType dataType;
            bool transA, transB, bias;
            int m, n, k;
            float alpha;
        } info;

        explicit MatMulCublas(decltype(info)) noexcept;

        static KernelBox build(bool transA, bool transB,
                               float alpha, float beta,
                               Tensor const &a,
                               Tensor const &b,
                               std::optional<std::reference_wrapper<Tensor const>> c) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower() const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CUDNN_KERNEL_HH
