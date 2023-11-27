#ifndef KERNEL_GATHER_H
#define KERNEL_GATHER_H

#include "../collector.h"

namespace refactor::kernel {

    struct GatherCollector final : public InfoCollector {
        uint32_t axis;

        constexpr GatherCollector(decltype(_target) target, uint32_t axis_) noexcept
            : InfoCollector(target), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_GATHER_H
