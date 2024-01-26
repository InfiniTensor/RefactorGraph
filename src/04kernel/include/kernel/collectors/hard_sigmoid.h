#ifndef KERNEL_HARD_SIGMOIG_H
#define KERNEL_HARD_SIGMOIG_H

#include "../collector.h"

namespace refactor::kernel {

    struct HardSigmoidCollector final : public InfoCollector {
        float alpha, beta;

        constexpr HardSigmoidCollector(decltype(_target) target, float alpha_, float beta_) noexcept
            : InfoCollector(target), alpha(alpha_), beta(beta_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };
}// namespace refactor::kernel

#endif// KERNEL_HARD_SIGMOIG_H

