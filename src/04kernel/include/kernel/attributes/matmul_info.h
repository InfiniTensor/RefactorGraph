#ifndef KERNEL_MATMUL_INFO_H
#define KERNEL_MATMUL_INFO_H

#include "kernel/attributes/broadcaster.h"
#include "kernel/attributes/expand_info.h"
#include <optional>

namespace refactor::kernel {

    struct MatMulInfo {
        DataType dataType;
        float alpha, beta;
        bool transA, transB;
        size_t m, k, n;
        std::optional<ExpandInfo> biasExpand;
        // A 2-directional broadcaster that deals with dimensions before the last 2 dimensions
        Broadcaster broadcaster;

        MatMulInfo(Tensor const &, Tensor const &,
                   std::optional<std::reference_wrapper<Tensor const>>,
                   bool, bool, float, float);
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_INFO_H
