#ifndef KERNEL_GATHER_H
#define KERNEL_GATHER_H

#include "../collector.h"
#include "../target.h"
#include "refactor/common.h"

namespace refactor::kernel {

    struct GatherCollector final : public InfoCollector {
        Target target;
        uint_lv2 axis;

        constexpr GatherCollector(Target target_, uint_lv2 axis_) noexcept
            : InfoCollector(), target(target_), axis(axis_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_GATHER_H
