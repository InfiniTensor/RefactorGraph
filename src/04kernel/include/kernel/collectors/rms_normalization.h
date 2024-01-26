#ifndef KERNEL_RMS_NORMALIZATION_H
#define KERNEL_RMS_NORMALIZATION_H

#include "../collector.h"

namespace refactor::kernel {

    struct RmsNormalizationCollector final : public InfoCollector {
        float epsilon;

        constexpr RmsNormalizationCollector(decltype(_target) target, float epsilon_) noexcept
            : InfoCollector(target), epsilon(epsilon_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_RMS_NORMALIZATION_H
