#ifndef KERNEL_SOFTMAX_H
#define KERNEL_SOFTMAX_H

#include "../collector.h"

namespace refactor::kernel {

    struct SoftmaxCollector final : public InfoCollector {
        dim_t axis;

        constexpr SoftmaxCollector(decltype(_target) target, dim_t axis_) noexcept
            : InfoCollector(target), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_H
