#ifndef KERNEL_REDUCE_H
#define KERNEL_REDUCE_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {
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
        std::vector<int64_t> axes;

        ReduceCollector(Target target_, ReduceType reduceType_, std::vector<int64_t> axes_) noexcept
            : InfoCollector(), target(target_), reduceType(reduceType_), axes(axes_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif// KERNEL_REDUCE_H
