#ifndef KERNEL_SOFTMAX_H
#define KERNEL_SOFTMAX_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct SoftmaxCollector final : public InfoCollector {
        Target target;
        dim_t axis;

        constexpr SoftmaxCollector(Target target_, dim_t axis_) noexcept
            : InfoCollector(), target(target_), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_H
