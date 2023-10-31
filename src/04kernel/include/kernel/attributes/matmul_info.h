#ifndef KERNEL_MATMUL_INFO_H
#define KERNEL_MATMUL_INFO_H

#include "kernel/attributes/broadcaster.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    enum BiasBroadcast {
        NONE, // default, element-wise or no bias
        CONST,// constant bias
        COL,  // column bias
        ROW,  // row bias
    };

    struct MatMulInfo {
        DataType dataType;
        float alpha, beta;
        bool transA, transB, useBias;
        size_t b, m, k, n;
        BiasBroadcast biasMode = BiasBroadcast::NONE;
        // A 2-directional broadcaster that deals with dimensions before the last 2 dimensions
        Broadcaster broadcaster;

        MatMulInfo(Tensor const &, Tensor const &, bool = false, bool = false,
                   float = 1.0f, float = 1.0f);

        MatMulInfo(Tensor const &, Tensor const &, Tensor const &, bool = false,
                   bool = false, float = 1.0f, float = 1.0f);
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_INFO_H
