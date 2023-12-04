#ifndef KERNEL_MATMUL_INFO_H
#define KERNEL_MATMUL_INFO_H

#include "kernel/attributes/broadcaster.h"
#include "kernel/attributes/expand_info.h"
#include <variant>

namespace refactor::kernel {

    struct MatMulInfo {
        DataType dataType;
        float alpha, beta;
        bool transA, transB;
        size_t m, k, n;
        // Expand operation info for biasd
        std::optional<ExpandInfo> biasExpand;
        // A constant batch or a 2-directional broadcaster that deals with dimensions before the last 2 dimensions
        std::variant<Broadcaster, size_t> broadcasterOrBatch;

        MatMulInfo(Tensor const &, Tensor const &,
                   std::optional<std::reference_wrapper<Tensor const>>,
                   bool, bool, float, float);
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_INFO_H
