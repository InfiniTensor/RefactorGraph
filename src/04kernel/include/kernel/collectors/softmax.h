#ifndef KERNEL_SOFTMAX_H
#define KERNEL_SOFTMAX_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct SoftmaxCollector final : public InfoCollector {
        Target target;
        uint_lv2 axis;

        constexpr SoftmaxCollector(Target target_, uint_lv2 axis_) noexcept
            : InfoCollector(), target(target_), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_H

