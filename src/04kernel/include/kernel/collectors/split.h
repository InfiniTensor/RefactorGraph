#ifndef KERNEL_SPLIT_H
#define KERNEL_SPLIT_H

#include "../collector.h"

namespace refactor::kernel {

    struct SplitCollector final : public InfoCollector {
        uint32_t axis;

        constexpr SplitCollector(decltype(_target) target, uint32_t axis_) noexcept
            : InfoCollector(target), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_H
