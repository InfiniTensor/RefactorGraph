#ifndef KERNEL_MATMUL_INFO_H
#define KERNEL_MATMUL_INFO_H

#include "kernel/attributes/broadcaster.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    enum class BiasType : uint8_t {
        NoBias,
        Scalar,
        RowVector,
        ColVector,
        Matrix,
    };

    struct MatMulInfo {
        DataType dataType;
        float alpha, beta;
        bool transA, transB;
        size_t m, k, n;
        BiasType biasType;
        // A 2-directional broadcaster that deals with dimensions before the last 2 dimensions
        Broadcaster broadcaster;

        MatMulInfo(Tensor const &, Tensor const &,
                   std::optional<std::reference_wrapper<Tensor const>>,
                   bool, bool, float, float);

        MatMulInfo(Tensor const &, Tensor const &, bool = false, bool = false,
                   float = 1.0f);
        MatMulInfo(Tensor const &, Tensor const &, Tensor const &, bool = false,
                   bool = false, float = 1.0f, float = 1.0f);

        size_t batch() const noexcept;
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_INFO_H
