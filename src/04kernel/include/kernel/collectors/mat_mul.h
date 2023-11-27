#ifndef KERNEL_MAT_MUL_H
#define KERNEL_MAT_MUL_H

#include "../collector.h"

namespace refactor::kernel {

    struct MatMulCollector final : public InfoCollector {
        float alpha, beta;
        bool transA, transB;

        constexpr MatMulCollector(decltype(_target) target, float alpha_, float beta_, bool transA_, bool transB_) noexcept
            : InfoCollector(target), alpha(alpha_), beta(beta_), transA(transA_), transB(transB_) {}

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_MAT_MUL_H
