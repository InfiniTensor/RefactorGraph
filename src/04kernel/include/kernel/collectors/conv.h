#ifndef KERNEL_CONV_H
#define KERNEL_CONV_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct ConvCollector final : public InfoCollector {
        Target target;

        constexpr ConvCollector(Target target_) noexcept
            : InfoCollector(), target(target_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_H
