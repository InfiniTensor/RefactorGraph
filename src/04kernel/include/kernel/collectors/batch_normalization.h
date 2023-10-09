#ifndef KERNEL_BATCH_NORMALIZATION_H
#define KERNEL_BATCH_NORMALIZATION_H

#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct BatchNormalizationCollector final : public InfoCollector {
        Target target;
        float epsilon;

        constexpr BatchNormalizationCollector(Target target_, float epsilon_) noexcept
            : InfoCollector(), target(target_), epsilon(epsilon_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_BATCH_NORMALIZATION_H
