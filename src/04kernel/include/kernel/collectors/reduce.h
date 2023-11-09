#ifndef KERNEL_REDUCE_H
#define KERNEL_REDUCE_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    using Axes = absl::InlinedVector<uint32_t, 4>;

    enum class ReduceType {
        Mean,
        L1,
        L2,
        LogSum,
        LogSumExp,
        Max,
        Min,
        Prod,
        Sum,
        SumSquare,
    };

    struct ReduceCollector final : public InfoCollector {
        Target target;
        ReduceType reduceType;
        Axes axes;

        ReduceCollector(Target, ReduceType, Axes) noexcept;

        std::vector<KernelBox> filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif// KERNEL_REDUCE_H
