#ifndef KERNEL_SPLIT_H
#define KERNEL_SPLIT_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct SplitCollector final : public InfoCollector {
        Target target;
        uint32_t axis;

        constexpr SplitCollector(Target target_, uint32_t axis_) noexcept
            : InfoCollector(), target(target_), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_SPLIT_H
